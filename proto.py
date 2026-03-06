# proto.py

import math, torch
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from finch.finch import FINCH
from utils.utils import check_and_fix_nan, move_to_device
import logging
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Public API
# =============================================================================

def _to_index_tensor(idx, device):
    """Convert indices (list/ndarray/tensor) to a long tensor on the target device."""
    if idx is None:
        return None
    if isinstance(idx, torch.Tensor):
        return idx.to(device=device, dtype=torch.long)
    return torch.as_tensor(idx, device=device, dtype=torch.long)


def generate_prototypes(node, args, cluster_method='mean'):
    """
    Generate class prototypes with optional sample-level DP clipping and VPP group clipping.
    """
    node.model.eval()
    feat_accum = {}

    # 1) Feature extraction
    with torch.no_grad():
        for x, y in node.train_loader:
            x = move_to_device(x, args.device)
            y = y.to(args.device)

            if hasattr(node.model, 'get_prototypes'):
                feat = node.model.get_prototypes(x)
            elif hasattr(node.model, 'get_features'):
                feat = node.model.get_features(x)
                if feat.dim() == 4:
                    feat = feat.mean(dim=(2, 3))
            elif args.method in ['mpft']:
                feat = node.model(x, return_backbone=True)[0]
            else:
                _, feat, _ = node.model(x, return_backbone=False)
                if feat.dim() == 4:
                    feat = feat.mean(dim=(2, 3))
                elif feat.dim() == 3:
                    feat = feat.mean(dim=1)

            for vec, lbl in zip(feat, y):
                feat_accum.setdefault(int(lbl.item()), []).append(vec.cpu())

    new_protos, cluster_sizes_dict = {}, {}

    # 2) VPP dimension partition (optional)
    device = args.device
    idx_A, idx_B = _prepare_vpp_partition(node, args, feat_accum, device)

    # 3) Per-class: DP clipping + clustering
    for lbl, feat_list in feat_accum.items():
        mat = torch.stack(feat_list, dim=0)  # [n_i, D] on CPU
        n_i, D = mat.shape
        mat = check_and_fix_nan(mat, f"Feature matrix for class {lbl}", replace_with_zero=True)

        # 3.1 Sample-level DP clipping (optionally group-wise via VPP)
        if args.privacy == 'dp' and hasattr(args, 'clip_proto_norm'):
            R = float(args.clip_proto_norm)
            use_group = (idx_A is not None) and (idx_B is not None) and (len(idx_A) > 0) and (len(idx_B) > 0)

            if not use_group:
                norms = mat.norm(p=2, dim=1, keepdim=True)
                factor = (R / (norms + 1e-12)).clamp(max=1.0)
                mat = mat * factor
                node._vpp_kappa_A = None
                node._vpp_kappa_B = None
            else:
                idx_A_cpu = _to_index_tensor(idx_A, device='cpu')
                idx_B_cpu = _to_index_tensor(idx_B, device='cpu')

                dA, dB = int(idx_A_cpu.numel()), int(idx_B_cpu.numel())
                d_total = dA + dB
                beta = float(getattr(args, "vpp_split_beta", 0.5))

                kappa_A_raw = (dA / (d_total + 1e-12)) ** beta
                kappa_B_raw = (dB / (d_total + 1e-12)) ** beta
                norm_factor = math.sqrt(kappa_A_raw ** 2 + kappa_B_raw ** 2 + 1e-12)
                kappa_A = kappa_A_raw / norm_factor
                kappa_B = kappa_B_raw / norm_factor

                r_A = R * kappa_A
                r_B = R * kappa_B

                zA = mat[:, idx_A_cpu]
                zB = mat[:, idx_B_cpu]
                nA = zA.norm(p=2, dim=1, keepdim=True)
                nB = zB.norm(p=2, dim=1, keepdim=True)
                zA = zA * (r_A / (nA + 1e-12)).clamp(max=1.0)
                zB = zB * (r_B / (nB + 1e-12)).clamp(max=1.0)
                mat[:, idx_A_cpu] = zA
                mat[:, idx_B_cpu] = zB

                node._vpp_kappa_A = float(kappa_A)
                node._vpp_kappa_B = float(kappa_B)

            feat_accum[lbl] = [mat[i] for i in range(mat.shape[0])]

        # 3.2 Clustering
        if cluster_method == 'mean':
            centers = mat.mean(0, keepdim=True)
            sizes = [n_i]

        elif cluster_method == 'kmeans':
            r = getattr(args, 'cluster_rate', 0.1)
            Ck = max(1, int(math.ceil(n_i * r)))
            if n_i <= Ck:
                centers, sizes = mat, [1] * n_i
            else:
                km = KMeans(n_clusters=Ck, random_state=0).fit(mat.numpy())
                centers = torch.from_numpy(km.cluster_centers_).float()
                labels = torch.as_tensor(km.labels_)
                sizes = [int((labels == k).sum()) for k in range(Ck)]

        elif cluster_method == 'finch':
            if n_i < 2:
                centers, sizes = mat, [1]
            else:
                clusters, num_clust, _ = FINCH(mat.numpy(), distance='cosine', verbose=False)
                idx = int(np.argmin(num_clust))
                labels = clusters[:, idx]
                uniq = np.unique(labels)
                groups = [mat.numpy()[labels == g] for g in uniq]
                centers = torch.from_numpy(np.stack([g.mean(0) for g in groups])).float()
                sizes = [int(g.shape[0]) for g in groups]

        elif cluster_method == 'random':
            r = getattr(args, 'cluster_rate', 1.0)
            Ck = max(1, int(math.ceil(n_i * r)))
            idxs = np.random.choice(n_i, size=Ck, replace=False) if Ck < n_i else np.arange(n_i)
            centers = mat[idxs]
            sizes = [1] * centers.shape[0]

        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")

        new_protos[lbl] = centers.to(args.device)
        cluster_sizes_dict[lbl] = sizes

    # 4) Base sensitivity per center: Δ_k = 2R / n_k
    sensitivity_dict = {}
    if args.privacy == 'dp' and hasattr(args, 'clip_proto_norm'):
        R = float(args.clip_proto_norm)
        for lbl, sizes in cluster_sizes_dict.items():
            sensitivity_dict[lbl] = [2.0 * R / max(1, n_k) for n_k in sizes]
    else:
        for lbl in cluster_sizes_dict.keys():
            sensitivity_dict[lbl] = None

    return new_protos, feat_accum, cluster_sizes_dict, sensitivity_dict


def add_dp_noise_to_prototypes(new_protos, feat_accum, cluster_sizes_dict, sensitivity_dict, args, node=None):
    device = next(iter(new_protos.values())).device if len(new_protos) > 0 else args.device
    noisy_protos = {}
    sigma = float(args.noise_multiplier)

    raw_idx_A = getattr(node, '_vpp_idx_A', None) if node else None
    raw_idx_B = getattr(node, '_vpp_idx_B', None) if node else None

    # VPP mode
    if args.noise_add == 'vpp' and (raw_idx_A is not None and raw_idx_B is not None):
        kappa_A = getattr(node, '_vpp_kappa_A', None)
        kappa_B = getattr(node, '_vpp_kappa_B', None)

        p = float(getattr(args, "vpp_weight_exp", 1.0))
        wA_raw, wB_raw = (kappa_A ** p), (kappa_B ** p)
        ws = max(1e-12, wA_raw + wB_raw)
        # NOTE: kept as-is (swapped) to preserve original behavior
        wA, wB = (wB_raw / ws), (wA_raw / ws)

        idx_A_dev = _to_index_tensor(raw_idx_A, device=device)
        idx_B_dev = _to_index_tensor(raw_idx_B, device=device)
        dA, dB = int(idx_A_dev.numel()), int(idx_B_dev.numel())

        for lbl, proto in new_protos.items():
            Ck, _ = proto.shape
            base_delta = torch.tensor([2.0 * float(args.clip_proto_norm) / max(1, n_k) for n_k in cluster_sizes_dict[lbl]], dtype=proto.dtype, device=device)

            delta_A = base_delta * float(kappa_A)
            delta_B = base_delta * float(kappa_B)

            sigma_A = sigma / math.sqrt(max(wA, 1e-12))
            sigma_B = sigma / math.sqrt(max(wB, 1e-12))

            std_A = delta_A * sigma_A
            std_B = delta_B * sigma_B

            noise = torch.zeros_like(proto)
            noise[:, idx_A_dev] = torch.randn((Ck, dA), device=device) * std_A.view(-1, 1)
            noise[:, idx_B_dev] = torch.randn((Ck, dB), device=device) * std_B.view(-1, 1)
            noisy_protos[lbl] = proto + noise

        return noisy_protos

    # Equal mode
    else:
        for lbl, proto in new_protos.items():
            Ck, _ = proto.shape
            sens = sensitivity_dict[lbl]
            sigma_vec = torch.tensor([sigma * s for s in sens], dtype=proto.dtype, device=device)
            noise = torch.randn_like(proto) * sigma_vec.view(-1, 1)
            noisy_protos[lbl] = proto + noise

        return noisy_protos


def save_dp_protos_to_node(node, dp_protos, cluster_sizes_dict):
    """Cache DP prototypes for potential use in subsequent rounds."""
    node.prev_dp_protos = {lbl: p.detach().clone() for lbl, p in dp_protos.items()}
    node.prev_cluster_sizes = {lbl: sizes[:] for lbl, sizes in cluster_sizes_dict.items()}


def save_vpp_partition_to_node(node):
    """Store the current round's VPP indices for cross-round stability evaluation."""
    if hasattr(node, '_vpp_idx_A') and node._vpp_idx_A is not None:
        if not hasattr(node, '_vpp_idx_A_history'):
            node._vpp_idx_A_history = []
        node._vpp_idx_A_history.append(node._vpp_idx_A.cpu().clone())


# =============================================================================
# VPP partition
# =============================================================================

def _prepare_vpp_partition(node,
                           args,
                           feat_accum: Dict[int, List[torch.Tensor]],
                           device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build and cache VPP dimension partition indices.
    """
    rho = float(getattr(args, 'vpp_rho', 0.3))
    rho = max(1e-6, min(0.49, rho))

    if len(feat_accum) == 0 or len(next(iter(feat_accum.values()))) == 0:
        return None, None
    d = feat_accum[next(iter(feat_accum.keys()))][0].numel()

    S, d = _vpp_score_from_samples(feat_accum, device=device)
    logging.info(f"[VPP][dp_topk] Score stats: max={S.max()}, min={S.min()}, mean={S.mean()}")

    eps_ratio = float(getattr(args, 'vpp_topk_eps_ratio', 0.0))
    total_eps = float(getattr(args, 'epsilon', 0.0))
    eps = total_eps * eps_ratio
    delta = float(getattr(args, 'vpp_topk_delta', 0.0))

    if eps <= 0.0:
        raise ValueError(f"dp_topk requires a positive budget. total_epsilon={total_eps}, vpp_topk_eps_ratio={eps_ratio}, eps_topk={eps}")

    idx_A, idx_B, lam, node._vpp_score_dp_topk = _dp_topk_partition_oneshot(S, rho=rho, eps=eps, delta=delta, clip_max=0.1)
    logging.info(f"[VPP][dp_topk] One-shot Laplace top-k: eps={eps}, delta={delta}, k={int(math.ceil(rho*d))}, lambda={lam:.4g}")

    node._vpp_idx_A, node._vpp_idx_B = idx_A, idx_B
    return idx_A, idx_B


def _dp_topk_partition_oneshot(
    S: torch.Tensor,
    rho: float,
    eps: float,
    delta: float,
    sf: Optional[float] = None,
    clip_max: Optional[float] = 1.0,
    clip_min: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    One-shot DP Top-k with score clipping.
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    if not (0.0 <= delta < 1.0):
        raise ValueError('delta must be in [0,1). Use delta==0 for pure DP.')

    if clip_max is not None:
        if clip_max <= clip_min:
            raise ValueError("clip_max must be > clip_min.")
        S = torch.clamp(S, min=clip_min, max=clip_max)
        if sf is None:
            sf = float(clip_max)

    if sf is None or sf <= 0.0:
        raise ValueError("Per-coordinate sensitivity sf must be > 0 (pass sf or enable clip_max).")

    m = int(S.numel())
    if m == 0:
        raise ValueError("Empty score vector.")
    k = max(1, min(m, int(math.ceil(float(rho) * m))))

    if delta == 0.0:
        lam = (2.0 * k * float(sf)) / float(eps)
    else:
        lam = (8.0 * float(sf) * math.sqrt(k * max(1.0, math.log(max(m, 2) / max(delta, 1e-12))))) / float(eps)

    lam = float(max(lam, 1e-12))

    device, dtype = S.device, S.dtype
    lap = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(lam, device=device, dtype=dtype),
    )
    noise = lap.rsample(S.shape)
    S_noisy = S + noise

    idx_A = torch.topk(S_noisy, k=k, largest=True).indices.to(torch.long)
    mask = torch.ones(m, dtype=torch.bool, device=device)
    mask[idx_A] = False
    idx_B = torch.arange(m, device=device, dtype=torch.long)[mask]

    return idx_A, idx_B, lam, S_noisy


# =============================================================================
# Discriminativeness score
# =============================================================================

def _vpp_score_from_samples(feat_accum: Dict[int, List[torch.Tensor]],
                            device: torch.device) -> Tuple[torch.Tensor, int]:
    """
    Compute a discriminativeness score per feature dimension using an ANOVA-style F-score.
    """
    any_lbl = next(iter(feat_accum.keys()))
    d = feat_accum[any_lbl][0].numel()
    Va = torch.zeros(d, device=device)
    Ve = torch.zeros(d, device=device)
    mu_c, N_c = {}, {}

    feats = {}
    for c, lst in feat_accum.items():
        Z = torch.stack(lst, dim=0).to(device)
        feats[c] = Z
        mu_c[c] = Z.mean(0)
        N_c[c] = Z.shape[0]

    C = len(feats)
    totN = sum(N_c.values())
    mu = sum((N_c[c] / totN) * mu_c[c] for c in feats.keys())

    for c, Z in feats.items():
        s2_c = ((Z - mu_c[c].view(1, -1)) ** 2).sum(dim=0) / max(N_c[c] - 1, 1)
        Va += (N_c[c] - 1) * s2_c
        Ve += N_c[c] * (mu_c[c] - mu) ** 2

    zeta = (torch.median(Ve) + 1e-12) * 1e-3
    S = (Ve / (C - 1)) / (Va / (totN - C) + zeta)

    S_max = S.max()
    if S_max > 1e-12:
        S = S / S_max

    return S, d