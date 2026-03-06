import numpy as np
import logging
import time
from sklearn.feature_selection import mutual_info_classif
import math
import torch 

def calibrate_sigma_prototype_rdp(num_releases: int,
                                   target_epsilon: float,
                                   target_delta: float,
                                   alphas: list = None,
                                   tol: float = 1e-3,
                                   max_sigma: float = 10000.0): 
    if alphas is None:
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]

    def rdp_to_epsilon(sigma):
        """计算给定σ下的(ε,δ)-DP保证"""
        if sigma <= 0:
            return float('inf')
        
        # 计算每个α下的RDP代价
        rdp_epsilons = []
        for alpha in alphas:
            # 单次发布的RDP: ε_α = α/(2σ²)
            rdp_single = alpha / (2 * sigma * sigma)
            # T轮串行合成
            rdp_total = num_releases * rdp_single
            # 转换为(ε,δ)
            epsilon_alpha = rdp_total + math.log(1.0 / target_delta) / (alpha - 1)
            rdp_epsilons.append(epsilon_alpha)
        
        # 取最小的ε
        return min(rdp_epsilons)
    
    # 检查max_sigma是否足够
    eps_at_max = rdp_to_epsilon(max_sigma)
    if eps_at_max > target_epsilon:
        raise ValueError(
            f"σ={max_sigma} 时 ε={eps_at_max:.2f} > 目标ε={target_epsilon}，"
            f"需要更大的max_sigma、放宽隐私预算，或增大RDP的α上界（alphas）"
        )
    
    # 二分查找最小的σ使得ε ≤ target_epsilon
    lo, hi = 1e-3, max_sigma
    
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        eps_mid = rdp_to_epsilon(mid)
        
        if eps_mid > target_epsilon:
            # 噪声不够，需要增大σ
            lo = mid
        else:
            # 噪声过多，尝试减小σ
            hi = mid
    
    return hi


@torch.no_grad()
def mi_dimension_ranking(feat_mat: torch.Tensor, labels: torch.Tensor, n_bins=10):
    """
    feat_mat : [N, D]  torch.float32  (on CPU)
    labels   : [N]     torch.long
    返回: numpy array, 维度 idx 按互信息降序
    """
    X = feat_mat.numpy()
    y = labels.numpy() 
    mi = mutual_info_classif(X, y, discrete_features=False, n_neighbors=3, random_state=0)
    return np.argsort(mi)[::-1]   # 降序


def add_dp_noise_to_prototypes(new_protos, feat_accum, cluster_sizes_dict, sensitivity_dict, args, client_sensitivity=None):
    """
    支持两级DP加噪策略:
      - 样本级（sample）DP: equal/vpp 按原型逐个加噪
      - 客户端级（client）DP: 对整个客户端的原型向量统一加噪

    new_protos: dict[label] -> Tensor[Ck, D] (已经是从裁剪后的样本生成的原型)
    feat_accum: dict[label] -> list of feature tensors (每个样本的 D 维特征，已裁剪)
    cluster_sizes_dict: dict[label] -> list[int]，每个中心的样本数 n_k
    sensitivity_dict: dict[label] -> list[float]，每个中心的敏感度 = 2*R/n_k（样本级）
    client_sensitivity: dict or None，客户端级敏感度信息（可选）
    args 需要字段:
        - dp_level: 'sample' 或 'client'
        - noise_multiplier (sigma)：基于RDP校准的噪声倍率
        - noise_add in {'equal','vpp'}（仅用于样本级DP）
        - vpp_rho (ρ) 用于两组划分的比例（仅用于vpp）
    
    修正说明：
    1. 样本裁剪已经在 generate_prototypes 中完成（聚类之前）
    2. 敏感度已经在 generate_prototypes 中计算为 Δ_k = 2R/n_k
    3. 加噪公式：noise_std = σ·Δ_k，满足记录级DP
    """
    device = next(iter(new_protos.values())).device if len(new_protos) > 0 else args.device
    noisy_protos = {}

    sigma = float(args.noise_multiplier)
    dp_level = getattr(args, 'dp_level', 'sample')
     
    # ==========================================
    # 样本级（记录级）DP加噪 - 逐原型加噪
    # ==========================================
    
    # =========================
    # 1) VPP：判别方差引导的两组自适应加噪
    # =========================
    if args.noise_add == 'vpp':
        # VPP 开始计时
        if hasattr(args, 'enable_timing_log') and args.enable_timing_log:
            t_vpp_start = time.time()
        
        # 开始计时方差计算
        if hasattr(args, 'enable_timing_log') and args.enable_timing_log:
            t_variance_start = time.time()
        
        labels = list(new_protos.keys())
        # 维度 D
        any_proto = next(iter(new_protos.values()))
        _, D = any_proto.shape
        zeta = 1e-6

        # 统计 Va（类内方差和）、Ve（类间离差和）
        # 注意：feat_accum中的样本在generate_prototypes中已经被裁剪过了
        Va = torch.zeros(D, device=device)
        n_c, mu_c = {}, {}
        total_n = 0

        for lbl in labels:
            feats_list = feat_accum.get(lbl, [])
            if len(feats_list) == 0:
                continue
            feats = torch.stack(feats_list, dim=0).to(device)   # [n_i, D]
            n_i = feats.shape[0]
            n_c[lbl] = n_i
            total_n += n_i
            mu_c[lbl] = feats.mean(dim=0)                       # [D]
            Va += feats.var(dim=0, unbiased=False)              # [D]

        # 若没有任何统计量，退回 equal
        if total_n == 0:
            for lbl, proto in new_protos.items():
                Ck, D = proto.shape
                # 使用预先计算的敏感度
                # 噪声标准差 = sigma * sensitivity
                sensitivities = sensitivity_dict[lbl]
                sigma_vec = torch.tensor(
                    [sigma * sens for sens in sensitivities],
                    dtype=proto.dtype, device=device
                )
                noise = torch.randn_like(proto) * sigma_vec.view(-1, 1)
                noisy_protos[lbl] = proto + noise
            return noisy_protos

        # 总均值
        mu = torch.zeros(D, device=device)
        for lbl in n_c:
            mu += (n_c[lbl] / total_n) * mu_c[lbl]

        # 类间离差和
        Ve = torch.zeros(D, device=device)
        for lbl in n_c:
            diff = (mu_c[lbl] - mu)
            Ve += n_c[lbl] * (diff * diff)

        # 判别分数 S
        S = Ve / (Va + zeta)  # [D]
        
        # 方差计算结束计时（存储到args临时变量，由调用方收集）
        if hasattr(args, 'enable_timing_log') and args.enable_timing_log:
            time_variance = time.time() - t_variance_start
            args._temp_variance_time = time_variance

        # 维度分组
        rho = float(getattr(args, 'vpp_rho', 0.3))
        rho = max(1e-6, min(0.49, rho))  # 建议 < 0.5
        dA = max(1, int(math.ceil(rho * D)))
        dB = D - dA

        idx_sorted = torch.argsort(S, descending=True)
        idx_A = idx_sorted[:dA]   # 判别子空间（少噪）
        idx_B = idx_sorted[dA:]   # 非判别子空间（多噪）

        # 分组噪声系数
        coef_A = math.sqrt(dA / D) if dA > 0 else 0.0
        coef_B = math.sqrt(dB / D) if dB > 0 else 0.0

        for lbl, proto in new_protos.items():
            flat = proto  # [Ck, D]
            Ck, _ = flat.shape

            # 使用预先计算的敏感度
            # 噪声标准差 = sigma * sensitivity
            sensitivities = sensitivity_dict[lbl]
            sigma_vec = torch.tensor(
                [sigma * sens for sens in sensitivities],
                dtype=flat.dtype, device=device
            )  # [Ck]

            # 分组噪声（直接加噪，不裁剪）
            noise = torch.zeros_like(flat)
            noise[:, idx_A] = torch.randn((Ck, dA), device=device) * (sigma_vec.view(-1, 1) * coef_A)
            noise[:, idx_B] = torch.randn((Ck, dB), device=device) * (sigma_vec.view(-1, 1) * coef_B)

            noisy_protos[lbl] = flat + noise

        # VPP 结束计时（存储到args临时变量）
        if hasattr(args, 'enable_timing_log') and args.enable_timing_log:
            time_vpp = time.time() - t_vpp_start
            args._temp_vpp_time = time_vpp

        return noisy_protos
    
    # =========================
    # 2) equal：等强度（行级）加噪
    # =========================
    if args.noise_add == 'equal':
        # Equal 开始计时
        if hasattr(args, 'enable_timing_log') and args.enable_timing_log:
            t_equal_start = time.time()
            # Equal加噪不需要计算方差，记录为0
            args._temp_variance_time = 0.0
        
        for lbl, proto in new_protos.items():
            flat = proto  # [Ck, D]
            Ck, D = flat.shape
            # 使用预先计算的敏感度
            # 噪声标准差 = sigma * sensitivity
            sensitivities = sensitivity_dict[lbl]
            sigma_vec = torch.tensor(
                [sigma * sens for sens in sensitivities],
                dtype=flat.dtype, device=device
            )
            noise = torch.randn_like(flat) * sigma_vec.view(-1, 1)
            noisy_protos[lbl] = flat + noise

        # Equal 结束计时（存储到args临时变量）
        if hasattr(args, 'enable_timing_log') and args.enable_timing_log:
            time_equal = time.time() - t_equal_start
            args._temp_equal_time = time_equal

        return noisy_protos


    # -------- 兜底：若传入了未支持的策略，按 equal 处理 --------
    for lbl, proto in new_protos.items():
        flat = proto  # [Ck, D]
        Ck, D = flat.shape
        # 使用预先计算的敏感度
        # 噪声标准差 = sigma * sensitivity
        sensitivities = sensitivity_dict[lbl]
        sigma_vec = torch.tensor(
            [sigma * sens for sens in sensitivities],
            dtype=flat.dtype, device=device
        )
        noise = torch.randn_like(flat) * sigma_vec.view(-1, 1)
        noisy_protos[lbl] = flat + noise

    return noisy_protos

 


