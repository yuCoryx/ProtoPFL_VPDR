# proto.py

import math
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from finch.finch import FINCH

from utils.tools import check_and_fix_nan, move_to_device, _to_index_tensor


def generate_prototypes(node, args, cluster_method='mean'):
    """
    Generate class prototypes with optional sample-level DP clipping and VPP group clipping.
    """
    node.model.eval()
    feat_accum = {}

    # reset current-round VPP cache
    node._vpp_idx_A = None
    node._vpp_idx_B = None
    node._vpp_kappa_A = None
    node._vpp_kappa_B = None
    node._vpp_topk_lambda = None
    node._vpp_score_dp_topk = None

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

            feat = check_and_fix_nan(feat, "feature extraction", replace_with_zero=True)

            for vec, lbl in zip(feat, y):
                feat_accum.setdefault(int(lbl.item()), []).append(vec.detach().cpu())

    new_protos, cluster_sizes_dict = {}, {}

    # 2) VPP dimension partition (optional, done once and cached on node)
    device = args.device
    idx_A, idx_B = _prepare_vpp_partition(node, args, feat_accum, device)

    # 3) Per-class: DP clipping + clustering
    for lbl, feat_list in feat_accum.items():
        mat = torch.stack(feat_list, dim=0) 
        n_i, D = mat.shape
        mat = check_and_fix_nan(mat, f"Feature matrix for class {lbl}", replace_with_zero=True)

        # 3.1 Sample-level DP clipping
        if hasattr(args, 'clip_proto_norm'):
            R = float(args.clip_proto_norm)
            use_group = (
                idx_A is not None and idx_B is not None
                and len(idx_A) > 0 and len(idx_B) > 0
            )

            if not use_group:
                norms = mat.norm(p=2, dim=1, keepdim=True)
                factor = (R / (norms + 1e-12)).clamp(max=1.0)
                mat = mat * factor
            else:
                idx_A_cpu = _to_index_tensor(idx_A, device='cpu')
                idx_B_cpu = _to_index_tensor(idx_B, device='cpu')

                dA, dB = int(idx_A_cpu.numel()), int(idx_B_cpu.numel())
                d_total = dA + dB

                # Paper-aligned parameter-free design:
                # kappa_A = sqrt(dA / d), kappa_B = sqrt(dB / d)
                kappa_A = math.sqrt(dA / max(d_total, 1))
                kappa_B = math.sqrt(dB / max(d_total, 1))

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

            feat_accum[lbl] = [mat[i].clone() for i in range(mat.shape[0])]

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

        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")

        new_protos[lbl] = centers.to(args.device)
        cluster_sizes_dict[lbl] = sizes

    # 4) Sensitivity per center
    sensitivity_dict = {}
    if hasattr(args, 'clip_proto_norm'):
        R = float(args.clip_proto_norm)

        use_group = (
            args.noise_add == 'vpp'
            and node._vpp_idx_A is not None
            and node._vpp_idx_B is not None
            and node._vpp_kappa_A is not None
            and node._vpp_kappa_B is not None
        )

        for lbl, sizes in cluster_sizes_dict.items():
            if not use_group:
                sensitivity_dict[lbl] = [2.0 * R / max(1, n_k) for n_k in sizes]
            else:
                kappa_A = float(node._vpp_kappa_A)
                kappa_B = float(node._vpp_kappa_B)
                sensitivity_dict[lbl] = []
                for n_k in sizes:
                    full = 2.0 * R / max(1, n_k)
                    sensitivity_dict[lbl].append({
                        "A": full * kappa_A,
                        "B": full * kappa_B,
                        "full": full,
                    })
    else:
        for lbl in cluster_sizes_dict.keys():
            sensitivity_dict[lbl] = None

    return new_protos, feat_accum, cluster_sizes_dict, sensitivity_dict


def save_vpp_partition_to_node(node):
    """Store the current round's VPP indices for cross-round stability evaluation."""
    if hasattr(node, '_vpp_idx_A') and node._vpp_idx_A is not None:
        if not hasattr(node, '_vpp_idx_A_history'):
            node._vpp_idx_A_history = []
        node._vpp_idx_A_history.append(node._vpp_idx_A.detach().cpu().clone())


# =============================================================================
# VPP partition
# =============================================================================

def _prepare_vpp_partition(
    node,
    args,
    feat_accum: Dict[int, List[torch.Tensor]],
    device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build and cache VPP dimension partition indices once per round.

    Returns:
        idx_A, idx_B
    """

    rho = float(getattr(args, 'vpp_rho', 0.3))
    rho = max(1e-6, min(0.5, rho))

    if len(feat_accum) == 0 or len(next(iter(feat_accum.values()))) == 0:
        return None, None

    S, d = _vpp_score_from_samples(feat_accum, device=device)

    eps_ratio = float(getattr(args, 'vpp_topk_eps_ratio', 0.0))
    total_eps = float(getattr(args, 'epsilon', 0.0))
    total_delta = float(getattr(args, 'delta', 0.0))
    delta_topk = float(getattr(args, 'vpp_topk_delta', 0.0))

    if eps_ratio <= 0.0:
        raise ValueError("VPP requires args.vpp_topk_eps_ratio > 0.")

    total_rounds = int(getattr(args, 'rounds', getattr(args, 'T', 1)))
    total_rounds = max(1, total_rounds)

    eps_topk_total = total_eps * eps_ratio
    eps_topk_round = eps_topk_total / total_rounds

    idx_A, idx_B, lam, noisy_scores = _dp_topk_partition_oneshot(
        S,
        rho=rho,
        eps=eps_topk_round,
        delta=delta_topk,
        clip_max=float(getattr(args, "vpp_score_clip_max", 0.1)),
        clip_min=0.0,
    )

    node._vpp_idx_A = idx_A.detach().clone()
    node._vpp_idx_B = idx_B.detach().clone()
    node._vpp_topk_lambda = float(lam)
    node._vpp_score_dp_topk = noisy_scores.detach().clone()

    return idx_A, idx_B


def _dp_topk_partition_oneshot(
    S: torch.Tensor,
    rho: float,
    eps: float,
    delta: float,
    sf: Optional[float] = None,
    clip_max: Optional[float] = 1.0,
    clip_min: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    One-shot DP Top-k with score clipping.

    Returns:
        idx_A, idx_B, lam, S_noisy
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    if not (0.0 <= delta < 1.0):
        raise ValueError("delta must be in [0,1).")

    if clip_max is not None:
        if clip_max <= clip_min:
            raise ValueError("clip_max must be > clip_min.")
        S = torch.clamp(S, min=clip_min, max=clip_max)
        if sf is None:
            sf = float(clip_max)

    if sf is None or sf <= 0.0:
        raise ValueError("Per-coordinate sensitivity sf must be > 0.")

    m = int(S.numel())
    if m == 0:
        raise ValueError("Empty score vector.")
    k = max(1, min(m, int(math.ceil(float(rho) * m))))

    # Pure DP Laplace top-k
    if delta == 0.0:
        lam = (2.0 * k * float(sf)) / float(eps)
    else:
        # Optional approximate-DP branch
        lam = (
            8.0 * float(sf) * math.sqrt(k * max(1.0, math.log(max(m, 2) / max(delta, 1e-12))))
        ) / float(eps)

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

def _vpp_score_from_samples(
    feat_accum: Dict[int, List[torch.Tensor]],
    device: torch.device
) -> Tuple[torch.Tensor, int]:
    """
    Compute a discriminativeness score per feature dimension using an ANOVA-style F-score.
    """
    any_lbl = next(iter(feat_accum.keys()))
    d = feat_accum[any_lbl][0].numel()

    between_ss = torch.zeros(d, device=device)
    within_ss = torch.zeros(d, device=device)
    mu_c, N_c = {}, {}

    feats = {}
    for c, lst in feat_accum.items():
        Z = torch.stack(lst, dim=0).to(device)
        Z = check_and_fix_nan(Z, f"vpp score feats class {c}", replace_with_zero=True)
        feats[c] = Z
        mu_c[c] = Z.mean(0)
        N_c[c] = Z.shape[0]

    C = len(feats)
    totN = sum(N_c.values())
    if C <= 1 or totN <= C:
        # degenerate case
        return torch.zeros(d, device=device), d

    mu = sum((N_c[c] / totN) * mu_c[c] for c in feats.keys())

    for c, Z in feats.items():
        s2_c = ((Z - mu_c[c].view(1, -1)) ** 2).sum(dim=0) / max(N_c[c] - 1, 1)
        within_ss += (N_c[c] - 1) * s2_c
        between_ss += N_c[c] * (mu_c[c] - mu) ** 2

    zeta = (torch.median(between_ss) + 1e-12) * 1e-3
    S = (between_ss / (C - 1)) / (within_ss / (totN - C) + zeta)

    return S, d


def add_dp_noise_to_prototypes(
    new_protos,
    sensitivity_dict,
    args,
    node=None,
):
    """
    Add DP Gaussian noise to prototypes.
    """
    device = next(iter(new_protos.values())).device if len(new_protos) > 0 else args.device
    noisy_protos = {}

    sigma_ref = float(args.noise_multiplier)

    if args.noise_add == 'vpp':
        if node is None:
            raise ValueError("VPP noise requires `node`.")

        idx_A = getattr(node, "_vpp_idx_A", None)
        idx_B = getattr(node, "_vpp_idx_B", None)
        kappa_A = getattr(node, "_vpp_kappa_A", None)
        kappa_B = getattr(node, "_vpp_kappa_B", None)

        if (
            idx_A is None or idx_B is None
            or kappa_A is None or kappa_B is None
            or len(idx_A) == 0 or len(idx_B) == 0
        ):
            raise ValueError("VPP partition/kappa is missing on node.")

        idx_A = _to_index_tensor(idx_A, device=device)
        idx_B = _to_index_tensor(idx_B, device=device)

        kappa_A = float(kappa_A)
        kappa_B = float(kappa_B)

        w_A = kappa_B / max(kappa_A + kappa_B, 1e-12)
        w_B = 1.0 - w_A

        sigma_A = sigma_ref / math.sqrt(max(w_A, 1e-12))
        sigma_B = sigma_ref / math.sqrt(max(w_B, 1e-12))

        for lbl, proto in new_protos.items():
            flat = proto
            Ck, _ = flat.shape
            sens_list = sensitivity_dict[lbl]

            noise = torch.zeros_like(flat)
            for i in range(Ck):
                sens_item = sens_list[i]
                delta_A = float(sens_item["A"])
                delta_B = float(sens_item["B"])

                if idx_A.numel() > 0:
                    noise[i, idx_A] = torch.randn(
                        idx_A.numel(), device=device, dtype=flat.dtype
                    ) * (sigma_A * delta_A)
                if idx_B.numel() > 0:
                    noise[i, idx_B] = torch.randn(
                        idx_B.numel(), device=device, dtype=flat.dtype
                    ) * (sigma_B * delta_B)

            noisy_protos[lbl] = flat + noise

        return noisy_protos

    elif args.noise_add == 'equal':
        for lbl, proto in new_protos.items():
            flat = proto
            sens_list = sensitivity_dict[lbl]
            sigma_vec = torch.tensor(
                [sigma_ref * float(sens) for sens in sens_list],
                dtype=flat.dtype,
                device=device,
            )
            noise = torch.randn_like(flat) * sigma_vec.view(-1, 1)
            noisy_protos[lbl] = flat + noise
        return noisy_protos

    else:
        raise ValueError(f"Unknown args.noise_add={args.noise_add}")