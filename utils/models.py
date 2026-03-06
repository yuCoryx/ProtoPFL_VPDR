# model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    from transformers import ViTModel
except Exception:
    ViTModel = None

try:
    from transformers import RobertaModel
except Exception:
    RobertaModel = None


class BaseFeatureExtractor(nn.Module):
    """Base class for all feature extractors used in heterogeneous settings."""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        raise NotImplementedError

    def get_feature_dim(self):
        return self.feature_dim


class ClassificationHead(nn.Module):
    """A configurable MLP classification head."""
    def __init__(self, input_dim, num_classes, hidden_dims=[512], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:  # [B, C, H, W]
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.dim() == 3:  # [B, T, D] or [B, D, T]
            x = x.mean(dim=1) if x.size(1) < x.size(2) else x.mean(dim=2)
        return self.classifier(x)


class BaseHeadModel(nn.Module):
    """
    Split model: feature_extractor + classification_head.

    forward returns:
        (proto_backbone, proto_adapter, logits)
    where proto_* are pooled to [B, K'].
    """
    def __init__(self, feature_extractor, classification_head, l2norm_proto=False):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        self.l2norm_proto = l2norm_proto

        self.adapter = getattr(self.feature_extractor, "adapter", nn.ModuleList()) or nn.ModuleList()
        self.classifier = self.classification_head

    def _pool_to_proto(self, feats):
        if feats.dim() == 4:  # [B, K', H, W]
            proto = feats.mean(dim=(2, 3))
        elif feats.dim() == 3:  # [B, T, K'] or [B, K', T]
            if feats.size(-1) == self.feature_extractor.get_feature_dim():
                proto = feats.mean(dim=1)
            elif feats.size(1) == self.feature_extractor.get_feature_dim():
                proto = feats.mean(dim=2)
            else:
                proto = feats.mean(dim=1)
        else:
            proto = feats

        return F.normalize(proto, p=2, dim=1) if self.l2norm_proto else proto

    def forward(self, x, return_backbone=False):
        result = self.feature_extractor(x, return_backbone=return_backbone)
        if isinstance(result, tuple):
            feats_backbone, feats_adapter = result
        else:
            feats_adapter = result
            feats_backbone = feats_adapter

        proto_backbone = self._pool_to_proto(feats_backbone)
        proto_adapter = self._pool_to_proto(feats_adapter)
        logits = self.classifier(proto_adapter)
        return proto_backbone, proto_adapter, logits

    def get_features(self, x):
        return self.feature_extractor(x)

    def get_prototypes(self, x, detach=True):
        with torch.no_grad():
            feats = self.feature_extractor(x)
            proto = self._pool_to_proto(feats)
            return proto.detach() if detach else proto


class LinearAdapter(nn.Module):
    """Residual MLP adapter for vector/token features."""
    def __init__(self, d_model, reduction=16, dropout=0.1):
        super().__init__()
        r = max(1, d_model // reduction)
        self.down = nn.Linear(d_model, r, bias=False)
        self.up = nn.Linear(r, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        h = F.relu(self.down(x))
        h = self.dropout(h)
        return x + self.up(h)


class ConvAdapter(nn.Module):
    """Residual 1x1 conv adapter for feature maps."""
    def __init__(self, channels, reduction=16, dropout=0.0):
        super().__init__()
        r = max(1, channels // reduction)
        self.down = nn.Conv2d(channels, r, 1, bias=False)
        self.up = nn.Conv2d(r, channels, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        h = F.relu(self.down(x))
        h = self.dropout(h)
        return x + self.up(h)


from torchvision.models.resnet import BasicBlock, Bottleneck


class BasicBlockWithAdapter(BasicBlock):
    """ResNet BasicBlock with a ConvAdapter inserted after conv2+bn2."""
    def __init__(self, orig: BasicBlock, reduction=16):
        super().__init__(orig.conv1.in_channels, orig.conv2.out_channels, stride=orig.stride, downsample=orig.downsample,
                         groups=getattr(orig, 'groups', 1), base_width=getattr(orig, 'base_width', 64),
                         dilation=getattr(orig, 'dilation', 1), norm_layer=orig.bn1.__class__)
        self.load_state_dict(orig.state_dict())
        self.adapter = ConvAdapter(self.bn2.num_features, reduction=reduction)

    def forward(self, x, return_backbone=False):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out_backbone = out
        out = self.adapter(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return (out, out_backbone) if return_backbone else out


class BottleneckWithAdapter(Bottleneck):
    """ResNet Bottleneck with a ConvAdapter inserted after conv3+bn3."""
    def __init__(self, orig: Bottleneck, reduction=16):
        super().__init__(inplanes=orig.conv1.in_channels, planes=orig.conv3.out_channels // self.expansion,
                         stride=orig.stride, downsample=orig.downsample, groups=getattr(orig, 'groups', 1),
                         base_width=getattr(orig, 'base_width', 64), dilation=getattr(orig, 'dilation', 1),
                         norm_layer=getattr(orig, 'bn1', None).__class__ if hasattr(orig, 'bn1') else None)
        self.load_state_dict(orig.state_dict())
        self.adapter = ConvAdapter(self.bn3.num_features, reduction=reduction)

    def forward(self, x, return_backbone=False):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        out_backbone = out
        out = self.adapter(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return (out, out_backbone) if return_backbone else out


def _patch_resnet_with_adapters(resnet: nn.Module, reduction=16):
    """Replace blocks in layer1..4 with adapter-augmented blocks and return all adapters."""
    adapters = nn.ModuleList()
    for lname in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(resnet, lname)
        for i, blk in enumerate(layer):
            if isinstance(blk, BasicBlock):
                blk_new = BasicBlockWithAdapter(blk, reduction=reduction)
            elif isinstance(blk, Bottleneck):
                blk_new = BottleneckWithAdapter(blk, reduction=reduction)
            else:
                blk_new = blk
            layer[i] = blk_new
            if hasattr(blk_new, 'adapter'):
                adapters.append(blk_new.adapter)
    return adapters


class ResNetFeatureExtractor(BaseFeatureExtractor):
    """ResNet backbone that outputs feature maps projected to K' channels."""
    def __init__(self, model_name='resnet18', feature_dim=512, pretrained=False, adapter_dim=64):
        super().__init__(feature_dim)

        name2ctor = {'resnet18': tv_models.resnet18, 'resnet34': tv_models.resnet34, 'resnet50': tv_models.resnet50,
                     'resnet101': tv_models.resnet101, 'resnet152': tv_models.resnet152}
        if model_name not in name2ctor:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        m = name2ctor[model_name](pretrained=pretrained)
        for p in m.parameters():
            p.requires_grad = False

        reduction = max(1, (512 if model_name in ['resnet18', 'resnet34'] else 2048) // max(1, adapter_dim))
        self.adapter = _patch_resnet_with_adapters(m, reduction=reduction)

        self.stem = nn.Sequential(*(list(m.children())[:4]))
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        in_ch = 512 if model_name in ['resnet18', 'resnet34'] else 2048
        self.proj = nn.Conv2d(in_ch, feature_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(feature_dim)

    def forward(self, x, return_backbone=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if return_backbone:
            x_backbone_list = []
            for blk in self.layer4:
                if hasattr(blk, 'adapter'):
                    x, x_bb = blk(x, return_backbone=True)
                    x_backbone_list.append(x_bb)
                else:
                    x = blk(x)
            x_backbone = x_backbone_list[-1] if x_backbone_list else x
            x_backbone = self.bn(self.proj(x_backbone))
            x_adapter = self.bn(self.proj(x))
            return x_backbone, x_adapter
        else:
            x = self.layer4(x)
            return self.bn(self.proj(x))


class ViTFeatureExtractor(BaseFeatureExtractor):
    """ViT backbone with a post-block LinearAdapter for each encoder layer."""
    def __init__(self, model_name='vit_tiny', feature_dim=512, pretrained=False, adapter_dim=64, return_token_seq=False, pool_type='patch_mean'):
        super().__init__(feature_dim)
        if ViTModel is None:
            raise ImportError("transformers is required: pip install transformers")

        local_root = os.path.join(os.path.dirname(__file__), 'model')
        name_to_dir = {'vit_tiny': 'vit-tiny', 'vit_small': 'vit-small', 'vit_base': 'vit-base-patch16-224-in21k'}
        local_dir = os.path.join(local_root, name_to_dir.get(model_name, 'vit-tiny'))
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Local ViT model directory not found: {local_dir}")

        self.backbone = ViTModel.from_pretrained(local_dir, local_files_only=True)
        embed_dim = getattr(self.backbone.config, 'hidden_size', 768)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.return_token_seq = return_token_seq
        self.pool_type = pool_type
        self.proj = nn.Linear(embed_dim, feature_dim)

        self.adapter = nn.ModuleList()
        red = max(1, embed_dim // adapter_dim)
        for _ in range(len(self.backbone.encoder.layer)):
            self.adapter.append(LinearAdapter(d_model=embed_dim, reduction=red))

    def forward(self, x, return_backbone=False):
        x = x.float()
        hidden = self.backbone.embeddings(x)

        for blk, adp in zip(self.backbone.encoder.layer, self.adapter):
            hidden_backbone = blk(hidden)[0]
            hidden = adp(hidden_backbone)

        hidden = self.backbone.layernorm(hidden)
        if return_backbone:
            hidden_backbone = self.backbone.layernorm(hidden_backbone)

        if self.return_token_seq:
            feat_adapter = self.proj(hidden[:, 1:, :])
            if return_backbone:
                feat_backbone = self.proj(hidden_backbone[:, 1:, :])
                return feat_backbone, feat_adapter
            return feat_adapter
        else:
            if self.pool_type == 'cls':
                feat_adapter = hidden[:, 0, :]
                if return_backbone:
                    feat_backbone = hidden_backbone[:, 0, :]
            else:
                feat_adapter = hidden[:, 1:, :].mean(dim=1)
                if return_backbone:
                    feat_backbone = hidden_backbone[:, 1:, :].mean(dim=1)

            feat_adapter = self.proj(feat_adapter)
            if return_backbone:
                feat_backbone = self.proj(feat_backbone)
                return feat_backbone, feat_adapter
            return feat_adapter


class RobertaFeatureExtractor(BaseFeatureExtractor):
    """RoBERTa backbone with a lightweight adapter and optional pooling."""
    def __init__(self, local_dir, feature_dim=512, adapter_dim=64, return_token_seq=True, pool_type="cls"):
        super().__init__(feature_dim)
        if RobertaModel is None:
            raise ImportError("transformers is required: pip install transformers")

        self.backbone = RobertaModel.from_pretrained(local_dir, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.return_token_seq = return_token_seq
        self.pool_type = pool_type

        red = max(1, hidden // max(1, adapter_dim))
        self.adapter = nn.ModuleList([LinearAdapter(d_model=hidden, reduction=red)])
        self.proj = nn.Linear(hidden, feature_dim)

    def forward(self, batch, return_backbone=False):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_backbone = out.last_hidden_state
        h_adapter = self.adapter[0](h_backbone)

        if self.return_token_seq:
            feat_adapter = self.proj(h_adapter)
            if return_backbone:
                feat_backbone = self.proj(h_backbone)
                return feat_backbone, feat_adapter
            return feat_adapter
        else:
            if self.pool_type == "cls":
                feat_adapter = self.proj(h_adapter[:, 0, :])
                if return_backbone:
                    feat_backbone = self.proj(h_backbone[:, 0, :])
                    return feat_backbone, feat_adapter
            else:
                feat_adapter = self.proj(h_adapter.mean(dim=1))
                if return_backbone:
                    feat_backbone = self.proj(h_backbone.mean(dim=1))
                    return feat_backbone, feat_adapter
            return feat_adapter


class ModelFactory:
    @staticmethod
    def create_feature_extractor(model_config):
        model_type = model_config['type']
        if model_type.startswith('resnet'):
            return ResNetFeatureExtractor(model_name=model_type, feature_dim=model_config.get('feature_dim', 512),
                                          pretrained=model_config.get('pretrained', False), adapter_dim=model_config.get('adapter_dim', 64))
        elif model_type.startswith('vit'):
            return ViTFeatureExtractor(model_name=model_type, feature_dim=model_config.get('feature_dim', 512),
                                       pretrained=model_config.get('pretrained', False), adapter_dim=model_config.get('adapter_dim', 64),
                                       return_token_seq=model_config.get('return_token_seq', False), pool_type=model_config.get('pool_type', 'patch_mean'))
        elif model_type.startswith('roberta'):
            return RobertaFeatureExtractor(local_dir=model_config.get('local_dir', os.path.join(os.path.dirname(__file__), 'model', 'roberta_base')),
                                           feature_dim=model_config.get('feature_dim', 512), adapter_dim=model_config.get('adapter_dim', 64),
                                           return_token_seq=model_config.get('return_token_seq', True), pool_type=model_config.get('pool_type', 'cls'))
        else:
            raise ValueError(f"Unsupported model type: {model_type}.")

    @staticmethod
    def create_classification_head(input_dim, num_classes, head_config):
        hidden_dims = head_config.get('hidden_dims', [512])
        dropout = head_config.get('dropout', 0.1)
        return ClassificationHead(input_dim, num_classes, hidden_dims, dropout=dropout)

    @staticmethod
    def create_heterogeneous_model(client_id, model_configs, num_classes, feature_dim=512, l2norm_proto=False):
        model_config = model_configs[client_id % len(model_configs)]
        feature_extractor = ModelFactory.create_feature_extractor(model_config)
        classification_head = ModelFactory.create_classification_head(feature_dim, num_classes, model_config.get('head', {}))
        return BaseHeadModel(feature_extractor, classification_head, l2norm_proto=l2norm_proto)


class TrainableGlobalPrototypes(nn.Module):
    """Learnable mapping from class id -> prototype embedding (FedTGP)."""
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embeddings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(nn.Linear(feature_dim, server_hidden_dim), nn.ReLU())
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        if isinstance(class_id, (list, tuple)):
            class_id = torch.tensor(class_id, device=self.device)
        elif not isinstance(class_id, torch.Tensor):
            class_id = torch.tensor([class_id], device=self.device)
        emb = self.embeddings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out


class ResNet18(nn.Module):
    """Compatibility wrapper for previous ResNet18 usage."""
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor('resnet18', feature_dim)
        self.classifier = ClassificationHead(feature_dim, num_classes)

    @property
    def adapter(self):
        return getattr(self.feature_extractor, "adapter", nn.ModuleList())

    def forward(self, x, return_backbone=False):
        result = self.feature_extractor(x, return_backbone=return_backbone)
        if isinstance(result, tuple):
            features_backbone, features_adapter = result
        else:
            features_adapter = result
            features_backbone = features_adapter

        if features_adapter.dim() == 4:
            feat_adapter = features_adapter.mean(dim=(2, 3))
            feat_backbone = features_backbone.mean(dim=(2, 3)) if features_backbone.dim() == 4 else features_backbone
        else:
            feat_adapter = features_adapter
            feat_backbone = features_backbone

        logits = self.classifier(feat_adapter)
        return feat_backbone, feat_adapter, logits


class ViT(nn.Module):
    def __init__(self, num_classes, model_name='vit_small', adapter_reduction=16, feature_dim=512, pool_type='patch_mean'):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor(model_name, feature_dim, adapter_dim=adapter_reduction, pool_type=pool_type)
        self.classifier = ClassificationHead(feature_dim, num_classes)

    @property
    def adapter(self):
        return getattr(self.feature_extractor, "adapter", nn.ModuleList())

    def forward(self, x, return_backbone=False):
        result = self.feature_extractor(x, return_backbone=return_backbone)
        if isinstance(result, tuple):
            feat_backbone, feat_adapter = result
        else:
            feat_adapter = result
            feat_backbone = feat_adapter
        logits = self.classifier(feat_adapter)
        return feat_backbone, feat_adapter, logits


class Roberta(nn.Module):
    """Compatibility wrapper for RoBERTa text batches."""
    def __init__(self, num_classes, model_dir='model/roberta_base', feature_dim=512, adapter_dim=64):
        super().__init__()
        self.feature_extractor = RobertaFeatureExtractor(local_dir=model_dir, feature_dim=feature_dim, adapter_dim=adapter_dim, return_token_seq=False, pool_type='cls')
        self.classifier = ClassificationHead(feature_dim, num_classes)

    @property
    def adapter(self):
        return getattr(self.feature_extractor, "adapter", nn.ModuleList())

    def forward(self, x, return_backbone=False):
        result = self.feature_extractor(x, return_backbone=return_backbone)
        if isinstance(result, tuple):
            feat_backbone, feat_adapter = result
        else:
            feat_adapter = result
            feat_backbone = feat_adapter
        logits = self.classifier(feat_adapter)
        return feat_backbone, feat_adapter, logits


HETEROGENEOUS_MODEL_CONFIGS = {
    'HtFE2': [
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'vit_tiny', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
    ],
    'HtFE4': [
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'resnet34', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'vit_tiny', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
        {'type': 'vit_small', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
    ],
    'HtFE6': [
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'resnet34', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'resnet50', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'vit_tiny', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
        {'type': 'vit_small', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
        {'type': 'vit_base', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
    ],
    'HtC4': [
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True, 'head': {'hidden_dims': [512]}},
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True, 'head': {'hidden_dims': [512, 256]}},
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True, 'head': {'hidden_dims': [512, 256, 128]}},
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True, 'head': {'hidden_dims': [1024, 512]}},
    ],
    'HtFE4-HtC2': [
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True, 'head': {'hidden_dims': [512]}},
        {'type': 'resnet34', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True, 'head': {'hidden_dims': [512, 256]}},
        {'type': 'vit_tiny', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True, 'head': {'hidden_dims': [512]}},
        {'type': 'vit_small', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True, 'head': {'hidden_dims': [512, 256]}},
    ],
    'ResNet4': [
        {'type': 'resnet18', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'resnet34', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'resnet50', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
        {'type': 'resnet101', 'feature_dim': 512, 'adapter_dim': 64, 'pretrained': True},
    ],
    'ViT4': [
        {'type': 'vit_tiny', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
        {'type': 'vit_small', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
        {'type': 'vit_base', 'feature_dim': 512, 'adapter_dim': 64, 'pool_type': 'patch_mean', 'pretrained': True},
        {'type': 'vit_base', 'feature_dim': 512, 'adapter_dim': 128, 'pool_type': 'patch_mean', 'pretrained': True},
    ],
}


