import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn


def setup_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def move_to_device(batch, device):
    """
    Move a batch (Tensor / dict / list/tuple) to the target device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    return batch


def get_model_info(model):
    """Return a compact summary of model size and structure."""
    if model is None:
        return "Model not initialized"

    info = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_type": type(model).__name__,
    }

    if hasattr(model, "feature_extractor"):
        info["feature_extractor"] = type(model.feature_extractor).__name__
        info["feature_dim"] = model.feature_extractor.get_feature_dim()

    if hasattr(model, "classification_head"):
        info["classification_head"] = type(model.classification_head).__name__

    return info

def freeze_layers(model: torch.nn.Module, unfreeze_layers: list[str]):
    """
    Freeze all parameters, then unfreeze parameters under selected module names.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    named_modules = dict(model.named_modules())
    for name in unfreeze_layers:
        module = named_modules.get(name, None)
        if module is None:
            if name == 'adapter' and hasattr(model, 'adapter'):
                module = model.adapter
            elif name == 'classifier' and hasattr(model, 'classifier'):
                module = model.classifier
            else:
                raise ValueError(f"Module '{name}' not found in the model.")
        for p in module.parameters():
            p.requires_grad_(True)


def calculate_infonce_loss(features: torch.Tensor, labels: torch.Tensor, global_prototypes: dict, temperature: float) -> torch.Tensor:
    """
    Multi-positive InfoNCE.
    """
    B, D = features.shape
    losses = []

    for i in range(B):
        f_i = features[i:i + 1]  # [1, D]
        y_i = labels[i].item()

        pos = global_prototypes[y_i].to(features.device)  # [P, D]
        neg_list = [global_prototypes[k] for k in global_prototypes if k != y_i]

        if len(neg_list) > 0:
            neg = torch.cat(neg_list, dim=0).to(features.device)  # [N, D]
            proto_all = torch.cat([pos, neg], dim=0)              # [P+N, D]
        else:
            proto_all = pos

        sim = F.cosine_similarity(f_i, proto_all, dim=1) / temperature
        exp_sim = torch.exp(sim)

        P = pos.size(0)
        mask = torch.zeros_like(exp_sim)
        mask[:P] = 1.0

        num = (exp_sim * mask).sum()
        den = exp_sim.sum()
        losses.append(-torch.log(num / den))

    return torch.stack(losses).mean()


def calculate_mse_loss(features: torch.Tensor, labels: torch.Tensor, global_prototypes: dict) -> torch.Tensor:
    """
    Compute MSE between features and class prototypes.
    """
    B, D = features.shape
    losses = []

    for i in range(B):
        cls = labels[i].item()
        if cls not in global_prototypes:
            continue

        f_i = features[i:i + 1]  # [1, D]
        pos = global_prototypes[cls].to(features.device)  # [P, D]
        diffs = (f_i - pos) ** 2
        losses.append(diffs.sum(dim=1).mean())

    return torch.stack(losses).mean()


def validate(args, node):
    """Standard classification evaluation using model logits."""
    model = node.model.to(args.device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in node.val_loader:
            x = move_to_device(x, args.device)
            y = y.to(args.device)
            _, _, logits = model(x)
            preds = logits.argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def validate_fedpcl(args, node):
    """
    Nearest-prototype evaluation for FedPCL using local prototypes.
    """
    model = node.model.to(args.device)
    model.eval()

    proto_dict = node.local_protos
    cls_list = sorted(proto_dict.keys())
    proto_list = []

    for c in cls_list:
        p = proto_dict[c].to(args.device)
        if p.dim() == 2:
            p = p.mean(dim=0)
        else:
            p = p.squeeze(0)
        proto_list.append(p)

    protos = torch.stack(proto_list, dim=0)
    protos = F.normalize(protos, dim=1)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in node.val_loader:
            x = move_to_device(x, args.device)
            y = y.to(args.device)

            _, feat, _ = model(x)
            feat = F.normalize(feat, dim=1)
            sims = feat @ protos.t()
            preds = sims.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def check_and_fix_nan(tensor: torch.Tensor, name: str = "tensor", replace_with_zero: bool = False) -> torch.Tensor:
    """
    Detect NaNs in a tensor and optionally replace them with zeros.
    """
    if torch.isnan(tensor).any():
        logging.warning(f"NaN detected in {name}.")
        if replace_with_zero:
            tensor = torch.nan_to_num(tensor, nan=0.0)
            logging.warning(f"Replaced NaNs with zeros in {name}.")
    return tensor