# init.py

import torch

from utils.utils import freeze_layers
from models import ModelFactory, HETEROGENEOUS_MODEL_CONFIGS, ResNet18, ViT, Roberta


class Node:
    """Container for client/server state in federated learning."""
    def __init__(self, num_id, train_loader=None, val_loader=None, num_classes=None, model=None, optimizer=None, args=None):
        self.num_id = num_id
        self.args = args
        self.node_num = args.node_num
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.model = model
        self.optimizer = optimizer

        self.sample_per_class = self.generate_sample_per_class() if self.train_loader is not None else None

        self.glob_proto = None
        self.glob_cluster_proto = None
        self.glob_unbiased_proto = None

        if self.args.method == 'feddyn':
            self.delta_c = [torch.zeros_like(p) for p in self.model.parameters()]

    def generate_sample_per_class(self):
        """Count samples per class in the local training set."""
        counts = torch.ones(self.num_classes, dtype=torch.long)
        for _, y in self.train_loader:
            for label in y:
                counts[label] += 1
        return counts


def init_model(args, num_classes):
    """Initialize either a homogeneous or heterogeneous model."""
    if getattr(args, "enable_heterogeneous", False):
        return init_heterogeneous_model(args, num_classes)
    return init_homogeneous_model(args, num_classes)


def init_homogeneous_model(args, num_classes):
    """Initialize a homogeneous model."""
    feature_dim = getattr(args, "feature_dim", 512)

    if args.model_type == "resnet18":
        model = ResNet18(num_classes, feature_dim=feature_dim)
    elif args.model_type == "vit_tiny":
        model = ViT(num_classes, model_name="vit_tiny", adapter_reduction=8, feature_dim=feature_dim)
    elif args.model_type == "vit_small":
        model = ViT(num_classes, model_name="vit_small", adapter_reduction=16, feature_dim=feature_dim)
    elif args.model_type == "vit_base":
        model = ViT(num_classes, model_name="vit_base", adapter_reduction=32, feature_dim=feature_dim)
    elif args.model_type == "roberta_base":
        model_dir = getattr(args, "model_dir", "model/roberta_base")
        model = Roberta(num_classes, model_dir=model_dir, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    freeze_layers(model, unfreeze_layers=["adapter"])
    return model


def _apply_heterogeneous_freeze_policy(model, args):
    """Apply the configured unfreeze policy for heterogeneous models."""
    if not hasattr(args, "unfreeze_layers"):
        return model

    if args.unfreeze_layers == "ad_cla":
        freeze_layers(model, unfreeze_layers=["feature_extractor.proj", "classification_head"])
    elif args.unfreeze_layers == "ad":
        freeze_layers(model, unfreeze_layers=["feature_extractor.proj"])

    return model


def init_heterogeneous_model(args, num_classes):
    """
    Initialize the server-side heterogeneous model using the first config
    in the selected model family.
    """
    model_family = getattr(args, "model_family", "HtFE4")
    feature_dim = getattr(args, "feature_dim", 512)

    if model_family not in HETEROGENEOUS_MODEL_CONFIGS:
        raise ValueError(f"Unsupported model family: {model_family}")

    model_configs = HETEROGENEOUS_MODEL_CONFIGS[model_family]
    model = ModelFactory.create_heterogeneous_model(0, model_configs, num_classes, feature_dim)
    return _apply_heterogeneous_freeze_policy(model, args)


def create_client_heterogeneous_model(client_id, args, num_classes):
    """Create a heterogeneous model for a specific client."""
    if not getattr(args, "enable_heterogeneous", False):
        return None

    model_family = getattr(args, "model_family", "HtFE4")
    feature_dim = getattr(args, "feature_dim", 512)

    if model_family not in HETEROGENEOUS_MODEL_CONFIGS:
        raise ValueError(f"Unsupported model family: {model_family}")

    model_configs = HETEROGENEOUS_MODEL_CONFIGS[model_family]
    model = ModelFactory.create_heterogeneous_model(client_id, model_configs, num_classes, feature_dim)
    return _apply_heterogeneous_freeze_policy(model, args)


def init_optimizer(params, optimizer_name, lr, weight_decay=5e-4):
    """Initialize an optimizer from either a module or a parameter iterable."""
    param_list = params.parameters() if hasattr(params, "parameters") else params

    if optimizer_name == "sgd":
        return torch.optim.SGD(param_list, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(param_list, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")
