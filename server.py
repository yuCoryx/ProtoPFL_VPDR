# server.py

import copy
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.init import init_optimizer
from utils.utils import freeze_layers
from utils.models import TrainableGlobalPrototypes


def receive_client_models(client_nodes, selected_indices):
    """
    Collect uploaded (local) prototypes from selected clients.
    """
    local_protos = {}
    for idx in selected_indices:
        node = client_nodes[idx]
        if hasattr(node, "local_protos") and node.local_protos is not None:
            local_protos[idx] = copy.deepcopy(node.local_protos)
    return local_protos


def Server_update(args, server_node, client_nodes, select_list):
    """
    Server-side update step
    """
    local_protos = receive_client_models(client_nodes, select_list)

    if args.method == "fedproto":
        server_node.glob_proto = copy.deepcopy(get_global_proto(local_protos, server_node, cluster_method="mean", args=args))

    elif args.method == "fedpcl":
        glob_proto = get_global_proto(local_protos, server_node, cluster_method="mean", args=args)
        server_node.glob_proto = copy.deepcopy(glob_proto)

        # Ensure each client has prototypes for all classes (fill missing with global ones)
        filled_local_protos = {}
        for cid, local in local_protos.items():
            client_protos = copy.deepcopy(local)
            for cls_label, proto in glob_proto.items():
                if cls_label not in client_protos:
                    client_protos[cls_label] = proto
            filled_local_protos[cid] = client_protos
        server_node.filled_local_protos = filled_local_protos

    elif args.method == "fedplvm":
        server_node.glob_proto = copy.deepcopy(get_global_proto(local_protos, server_node, cluster_method="finch", args=args))

    elif args.method == "fpl":
        glob_cluster_proto = get_global_proto(local_protos, server_node, cluster_method="finch", args=args)
        server_node.glob_cluster_proto = copy.deepcopy(glob_cluster_proto)

        # Unbiased prototype = mean over cluster centers
        server_node.glob_unbiased_proto = {c: centers.mean(dim=0, keepdim=True) for c, centers in glob_cluster_proto.items()}

    elif args.method == "fedtgp":
        update_fedtgp_server(args, server_node, select_list, local_protos)

    elif args.method == "mpft":
        prototypes, labels = get_proto_dataset(args, local_protos)
        mpft_server_finetune(args, server_node, prototypes, labels)

    else:
        raise ValueError(f"Undefined server method: {args.method}")

    return server_node


def get_global_proto(local_protos, server_node, cluster_method, args):
    """
    Aggregate prototypes per class across clients, with optional clustering.
    """
    protos_by_class = {cls: [] for cls in range(server_node.num_classes)}
    client_centers = {cls: {} for cls in protos_by_class}

    # Collect prototypes
    for cid, proto_dict in local_protos.items():
        for cls, p in proto_dict.items():
            p = p.to(args.device)
            if p.dim() == 1:
                p = p.unsqueeze(0)  # [D] -> [1, D]
            protos_by_class[cls].append(p)
            client_centers[cls][cid] = p

    total_protos = sum(len(lst) for lst in protos_by_class.values())
    n_clusters = max(1, int(args.cluster_rate * total_protos))

    global_proto = {}
    for cls, proto_list in protos_by_class.items():
        if not proto_list:
            continue

        class_protos = torch.cat(proto_list, dim=0)  # [N_cls, D]
        D = class_protos.shape[1]
        data_np = class_protos.detach().cpu().numpy()

        # Mean aggregation (optionally weighted for FedProto)
        if cluster_method == "mean" or n_clusters == 1:
            if args.method == "fedproto":
                sample_sizes = args.sample_sizes
                total_size = sum(sample_sizes)
                weighted = torch.zeros(D, device=args.device)
                for cid, centers in client_centers[cls].items():
                    local_mean = centers.mean(dim=0)
                    w = sample_sizes[cid] / total_size
                    weighted += w * local_mean
                global_proto[cls] = weighted.unsqueeze(0)
            else:
                global_proto[cls] = class_protos.mean(dim=0, keepdim=True)
            continue

        # Clustering aggregation
        if cluster_method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(data_np)
            centers_np = kmeans.cluster_centers_

        elif cluster_method == "finch":
            from finch.finch import FINCH
            c, num_clust, _ = FINCH(data_np, req_clust=None, distance="cosine", verbose=False)
            sel = int(np.argmin(num_clust))
            labels = c[:, sel]
            centers_np = np.stack([data_np[labels == lab].mean(axis=0) for lab in np.unique(labels)], axis=0)

        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")

        global_proto[cls] = torch.tensor(centers_np, dtype=torch.float32, device=args.device)

    actual_centers = sum(v.shape[0] for v in global_proto.values())
    logging.info(f"[Server] Global prototype aggregation done: method={cluster_method}, total_centers={actual_centers}.")
    return global_proto


def get_proto_dataset(args, local_protos):
    """
    Flatten all uploaded prototypes into a supervised dataset for MPFT server fine-tuning.
    """
    protos_all = []
    labels_all = []

    for _, proto_dict in local_protos.items():
        for lbl, proto_tensor in proto_dict.items():
            protos_all.append(proto_tensor.to(args.device))
            labels_all.extend([lbl] * proto_tensor.size(0))

    prototypes = torch.cat(protos_all, dim=0)
    labels = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    if prototypes.numel() == 0:
        raise ValueError("No prototypes collected for MPFT.")

    logging.info(f"[MPFT] prototypes.shape={prototypes.shape}, labels.shape={labels.shape}")
    return prototypes, labels


def mpft_server_finetune(args, server_node, prototypes: torch.Tensor, labels: torch.Tensor):
    """
    MPFT server-side fine-tuning using uploaded prototypes.
    """
    freeze_layers(server_node.model, unfreeze_layers=["adapter", "classifier"])

    server_node.optimizer = init_optimizer(server_node.model, "sgd", 1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(server_node.optimizer, milestones=[20, 40], gamma=0.1)
    server_node.model.train()

    # Extra adapter for prototype inputs (if not already initialized)
    if not hasattr(server_node, "proto_adapter"):
        from models import LinearAdapter
        server_node.proto_adapter = LinearAdapter(d_model=args.feature_dim, reduction=8, dropout=0.1).to(args.device)
        server_node.optimizer.add_param_group({"params": server_node.proto_adapter.parameters()})

    loader = DataLoader(TensorDataset(prototypes, labels), batch_size=64, shuffle=True)

    for epoch in range(args.mpft_sepoch):
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch_protos, batch_lbls in loader:
            batch_protos = batch_protos.to(args.device)
            batch_lbls = batch_lbls.to(args.device)

            server_node.optimizer.zero_grad()
            feat_adapter = server_node.proto_adapter(batch_protos)
            logits = server_node.model.classifier(feat_adapter)

            loss = F.cross_entropy(logits, batch_lbls)
            loss.backward()
            server_node.optimizer.step()

            bs = batch_lbls.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == batch_lbls).sum().item()
            total_samples += bs

        scheduler.step()
        logging.info(f"[MPFT Server] SE {epoch+1}/{args.mpft_sepoch} | train_loss={total_loss/total_samples:.4f} | train_acc={100.0*total_correct/total_samples:.2f}%")


def update_fedtgp_server(args, server_node, select_list, local_protos):
    """
    FedTGP server update 
    """
    if not hasattr(server_node, "tgp_model"):
        server_node.tgp_model = TrainableGlobalPrototypes(
            num_classes=args.num_classes,
            server_hidden_dim=args.feature_dim,
            feature_dim=args.feature_dim,
            device=args.device
        ).to(args.device)
 
    uploaded = []
    for cid in select_list:
        if cid not in local_protos:
            continue
        for cls_label, proto in local_protos[cid].items():
            p = proto.detach()
            if p.dim() == 2:
                p = p.mean(dim=0) if p.size(0) > 1 else p.squeeze(0)
            elif p.dim() > 2:
                p = p.view(-1, p.shape[-1]).mean(dim=0)
            uploaded.append((p.cpu(), int(cls_label)))

    if not uploaded:
        logging.warning("[FedTGP] No prototypes received from clients.")
        return
 
    avg_protos = proto_cluster([local_protos[cid] for cid in select_list if cid in local_protos])
    avg_protos = {k: v.to(args.device) for k, v in avg_protos.items()}

    gap = torch.ones(args.num_classes, device=args.device) * 1e9
    for k1 in avg_protos.keys():
        for k2 in avg_protos.keys():
            if k1 > k2 and avg_protos[k1].shape == avg_protos[k2].shape:
                dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                gap[k1] = torch.min(gap[k1], dis)
                gap[k2] = torch.min(gap[k2], dis)

    min_gap = torch.min(gap)
    gap = torch.where(gap > 1e8, min_gap, gap)
    max_gap = torch.max(gap) 

    protos_tensor = torch.stack([p for p, _ in uploaded], dim=0)          # [N, D]
    labels_tensor = torch.tensor([y for _, y in uploaded], dtype=torch.long)  # [N]
    proto_loader = DataLoader(TensorDataset(protos_tensor, labels_tensor), batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(server_node.tgp_model.parameters(), lr=args.lr)
    server_node.tgp_model.train()

    for _ in range(args.fedtgp_server_epochs):
        for proto, y in proto_loader:
            proto = proto.to(args.device)
            y = y.to(args.device)

            class_ids = torch.arange(args.num_classes, device=args.device)
            proto_gen = server_node.tgp_model(class_ids)  # [C, D]

            # Pairwise distance: [B, C]
            features_square = (proto ** 2).sum(dim=1, keepdim=True)
            centers_square = (proto_gen ** 2).sum(dim=1, keepdim=True)
            dist_sq = features_square - 2.0 * (proto @ proto_gen.t()) + centers_square.t()
            dist = torch.sqrt(dist_sq.clamp_min(1e-12))

            # Add per-class margin on the true class
            one_hot = F.one_hot(y, num_classes=args.num_classes).float()
            margin = min(max_gap.item(), args.fedtgp_margin_threshold)
            dist = dist + one_hot * margin

            loss = F.cross_entropy(-dist, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Export the final global prototypes (one per class)
    server_node.tgp_model.eval()
    with torch.no_grad():
        server_node.glob_proto = {c: server_node.tgp_model(torch.tensor(c, device=args.device)).detach() for c in range(args.num_classes)}


def proto_cluster(protos_list):
    """
    FedTGP-style per-class averaging over uploaded prototypes. 
    """
    buckets = defaultdict(list)
    for protos in protos_list:
        for k, v in protos.items():
            buckets[k].append(v)

    result = {}
    for k, items in buckets.items():
        shapes = [t.shape for t in items]
        if len(set(shapes)) == 1:
            s = torch.zeros_like(items[0])
            for t in items:
                s += t
            result[k] = (s / len(items)).detach()
        else:
            result[k] = items[0].detach()
    return result