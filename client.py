# client.py

import copy
import logging
import torch
import torch.nn.functional as F

from utils.init import Node
from utils.utils import freeze_layers, calculate_infonce_loss, check_and_fix_nan, move_to_device
from utils.dp_utils import add_dp_noise_to_prototypes
from proto import generate_prototypes


# =============================================================================
# DCR
# =============================================================================
 
def soft_clip(f, R, tau=0.01):
    """
    Continuous differentiable approximation of L2 clipping.
    """
    norm = f.norm(p=2, dim=1, keepdim=True) + 1e-12
    scale = R / (norm + tau * R)
    return f * torch.minimum(scale, torch.ones_like(scale))


def ensure_ema_teacher_heads(model):
    """Attach a frozen EMA teacher head (classifier_t) if missing."""
    if hasattr(model, "classifier_t"):
        return model
    if not hasattr(model, "classifier"):
        raise AttributeError("Model must have `classifier` to create EMA teacher head.")
    model.classifier_t = copy.deepcopy(model.classifier)
    for p in model.classifier_t.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def ema_update_teacher(model, m: float = 0.999):
    """EMA update: classifier_t <- m*classifier_t + (1-m)*classifier."""
    if not (hasattr(model, "classifier") and hasattr(model, "classifier_t")):
        raise AttributeError("Model must have `classifier` and `classifier_t`.")
    for ps, pt in zip(model.classifier.parameters(), model.classifier_t.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        

def ensure_dcr_state(node, args):
    """
    Ensure EMA teacher head and local step counter exist when DCR is enabled.
    """
    if not args.use_dcr:
        return
    node.model = ensure_ema_teacher_heads(node.model)
    if not hasattr(node, "_local_step"):
        node._local_step = 0
    node.model.classifier_t.eval()


def dcr_forward(node, args, x):
    """
    Standard forward used by multiple methods.

    Returns:
        feat_for_loss: clipped feature if DCR else raw feature
        logits_s: student logits computed on feat_for_loss
        logits_t: EMA-teacher logits computed on raw feature (None if not DCR)
        feat_raw: raw feature (adapter feature from model)
    """
    _, feat, _ = node.model(x)
    feat_raw = feat

    if args.use_dcr:
        feat_c = soft_clip(feat_raw, args.clip_proto_norm, args.softclip_tau)
        logits_s = node.model.classifier(feat_c)
        with torch.no_grad():
            logits_t = node.model.classifier_t(feat_raw)
        return feat_c, logits_s, logits_t, feat_raw

    logits_s = node.model.classifier(feat_raw)
    return feat_raw, logits_s, None, feat_raw


def compute_dcr_kd(node, args, logits_s, logits_t):
    """
    DCR KD: EMA-teacher logits_t -> student logits_s (with warmup).
    """
    if (not args.use_dcr) or (logits_t is None):
        return torch.tensor(0.0, device=logits_s.device), 0.0, False

    kd_warm = getattr(args, "kd_warmup_steps", 0)
    do_kd = (node._local_step >= kd_warm) and (args.dcr_kd_weight > 0.0)
    if not do_kd:
        return torch.tensor(0.0, device=logits_s.device), 0.0, False

    T = args.dcr_kd_T
    kd = F.kl_div(
        F.log_softmax(logits_s / T, dim=1),
        F.softmax(logits_t / T, dim=1),
        reduction="batchmean"
    ) * (T * T)
    return kd, float(args.dcr_kd_weight), True


def train_loop(node, args, batch_step_fn):
    """
    Shared per-epoch training loop.

    batch_step_fn(x, y) must return:
        loss (Tensor), logits_for_acc (Tensor), extras (dict), do_kd (bool)
    """
    node.model.train()
    ensure_dcr_state(node, args)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    sums = {}

    for x, y in node.train_loader:
        x = move_to_device(x, args.device)
        y = y.to(args.device)

        node.optimizer.zero_grad()
        loss, logits_acc, extras, do_kd = batch_step_fn(x, y)
        loss.backward()
        node.optimizer.step()

        # EMA update only when DCR KD is actually applied
        if args.use_dcr and do_kd:
            ema_update_teacher(node.model, m=getattr(args, "ema_m", 0.999))

        if args.use_dcr:
            node._local_step += 1

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits_acc.argmax(dim=1) == y).sum().item()
        total_samples += bs

        for k, v in extras.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = 100.0 * total_correct / max(1, total_samples)
    avg_extras = {k: v / max(1, total_samples) for k, v in sums.items()}
    return avg_loss, avg_acc, avg_extras


# =============================================================================
# Client encode / update
# =============================================================================

def Client_encode(args, client_nodes: dict, select_list: list, flag=None, round_idx=None):
    """
    Generate client prototypes and optionally add DP noise before upload.
    FedPCL validation refreshes local prototypes without DP noise.
    """
    method = args.method

    if method == "fedpcl" and flag == "validation":
        for idx in select_list:
            node = client_nodes[idx]
            raw_protos, feat_accum, cluster_sizes, sens = generate_prototypes(node, args, cluster_method="mean")
            node.local_protos = raw_protos
        logging.info("Validation prototypes updated.")
        return

    if method in ["fedproto", "fedpcl", "fedtgp", "mpft", "fpl"]:
        cluster_method = "mean"
    elif method in ["fedplvm"]:
        cluster_method = "finch"
    else:
        cluster_method = args.cluster_method

    for idx in select_list:
        node = client_nodes[idx]
        raw_protos, feat_accum, cluster_sizes, sens = generate_prototypes(node, args, cluster_method=cluster_method)
        logging.info(f"[Local][{method.upper()}][Client {idx}] Prototype generation completed.")

        if args.privacy == "dp":
            noised_protos = add_dp_noise_to_prototypes(raw_protos, feat_accum, cluster_sizes, sens, args, node)
            logging.info(f"[Local][{method.upper()}][Client {idx}] Applied DP noise: {args.noise_add}")
        else:
            noised_protos = raw_protos

        node.raw_protos = raw_protos
        node.noised_protos = noised_protos
        node.local_protos = noised_protos


def receive_server_model(args, clients: dict, server: Node, rnd):
    """Broadcast server state to clients based on method."""
    for node in clients.values():
        if args.method in ["mpft"]:
            teacher = copy.deepcopy(server.model).to(args.device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            node.teacher_model = teacher

        elif args.method in ["fedpcl"]:
            node.glob_proto = copy.deepcopy(server.glob_proto)
            node.all_local_protos = copy.deepcopy(server.filled_local_protos)

        elif args.method in ["fpl"]:
            node.glob_cluster_proto = copy.deepcopy(server.glob_cluster_proto)
            node.glob_unbiased_proto = copy.deepcopy(server.glob_unbiased_proto)

        else:
            node.glob_proto = copy.deepcopy(server.glob_proto)

    return clients


def Client_update(args, rnd, client_nodes: dict, server: Node, select_list: list):
    """
    Run local training on selected clients (adapter + classifier only).
    """
    client_nodes = receive_server_model(args, client_nodes, server, rnd)

    losses, accs = [], []

    for idx in select_list:
        node = client_nodes[idx]
        freeze_layers(node.model, unfreeze_layers=["adapter", "classifier"])

        loss, acc = 0.0, 0.0

        for epoch in range(args.E):
            if args.method == "fedproto":
                out = train_fedproto(node, args)
                if len(out) == 5:
                    loss, ce, mse, kd, acc = out
                    logging.info(f"[Local][FEDPROTO] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} MSE={mse:.4f} KD={kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")
                else:
                    loss, ce, mse, acc = out
                    logging.info(f"[Local][FEDPROTO] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} MSE={mse:.4f} Total={loss:.4f} Acc={acc:.2f}%")

            elif args.method == "fedpcl":
                out = train_fedpcl(node, args)
                if len(out) == 5:
                    loss, Lg, Lp, kd, acc = out
                    logging.info(f"[Local][FEDPCL] C{idx} E{epoch+1}/{args.E} Lg={Lg:.4f} Lp={Lp:.4f} KD={kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")
                else:
                    loss, Lg, Lp, acc = out
                    logging.info(f"[Local][FEDPCL] C{idx} E{epoch+1}/{args.E} Lg={Lg:.4f} Lp={Lp:.4f} Total={loss:.4f} Acc={acc:.2f}%")

            elif args.method == "fedplvm":
                out = train_fedplvm(node, args)
                if len(out) == 6:
                    loss, ce, contra, corr, kd, acc = out
                    logging.info(f"[Local][FEDPLVM] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} Contra={contra:.4f} Corr={corr:.4f} KD={kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")
                else:
                    loss, ce, contra, corr, acc = out
                    logging.info(f"[Local][FEDPLVM] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} Contra={contra:.4f} Corr={corr:.4f} Total={loss:.4f} Acc={acc:.2f}%")

            elif args.method == "fpl":
                loss, ce, kd, contra, mse, acc = train_fpl(node, args)
                if kd is not None:
                    logging.info(f"[Local][FPL] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} Contra={contra:.4f} MSE={mse:.4f} KD={kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")
                else:
                    logging.info(f"[Local][FPL] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} Contra={contra:.4f} MSE={mse:.4f} Total={loss:.4f} Acc={acc:.2f}%")

            elif args.method == "fedtgp":
                out = train_fedtgp(node, args)
                if len(out) == 5:
                    loss, ce, preg, kd, acc = out
                    logging.info(f"[Local][FEDTGP] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} ProtoReg={preg:.4f} KD={kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")
                else:
                    loss, ce, preg, acc = out
                    logging.info(f"[Local][FEDTGP] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} ProtoReg={preg:.4f} Total={loss:.4f} Acc={acc:.2f}%")

            elif args.method == "mpft":
                out = train_mpft(node, args)
                if len(out) == 5:
                    loss, ce, kd, dcr_kd, acc = out
                    logging.info(f"[Local][MPFT] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} KD={kd:.4f} DCR_KD={dcr_kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")
                else:
                    loss, ce, kd, acc = out
                    logging.info(f"[Local][MPFT] C{idx} E{epoch+1}/{args.E} CE={ce:.4f} KD={kd:.4f} Total={loss:.4f} Acc={acc:.2f}%")

        losses.append(loss)
        accs.append(acc)
        logging.info("--" * 30)

    return client_nodes, (sum(losses) / len(losses) if losses else 0.0), (sum(accs) / len(accs) if accs else 0.0)


# =============================================================================
# Train functions
# =============================================================================

def train_fedproto(node: Node, args):
    """
    FedProto: CE + proto MSE + optional DCR KD.
    """
    def step(x, y):
        feat, logits_s, logits_t, _ = dcr_forward(node, args, x)
        ce = F.cross_entropy(logits_s, y)

        if node.glob_proto is not None:
            labels_in_batch = [int(lbl) for lbl in y]
            if all(lbl in node.glob_proto for lbl in labels_in_batch):
                proto_vec = torch.stack([node.glob_proto[lbl] for lbl in labels_in_batch])
                mse = args.fedproto_mse_weight * F.mse_loss(feat, proto_vec)
            else:
                mse = torch.tensor(0.0, device=args.device)
        else:
            mse = torch.tensor(0.0, device=args.device)

        kd, kd_w, do_kd = compute_dcr_kd(node, args, logits_s, logits_t)
        loss = ce + mse + kd_w * kd
        extras = {"ce": ce.item(), "mse": mse.item(), "kd": kd.item()}
        return loss, logits_s, extras, do_kd

    avg_loss, avg_acc, ex = train_loop(node, args, step)
    if args.use_dcr:
        return avg_loss, ex["ce"], ex["mse"], ex["kd"], avg_acc
    return avg_loss, ex["ce"], ex["mse"], avg_acc


def train_fedpcl(node, args):
    """
    FedPCL: Lg + Lp + optional DCR KD.
    """
    device = args.device
    criterion = torch.nn.CrossEntropyLoss()
    tau = args.fedpcl_tau

    cls_list = sorted(node.glob_proto.keys())
    global_protos = torch.stack([node.glob_proto[c].squeeze(0).to(device) for c in cls_list], dim=0)
    global_protos = F.normalize(global_protos, dim=1)

    client_proto_sets = []
    for cid in node.all_local_protos.keys():
        cp = torch.stack([node.all_local_protos[cid][c].squeeze(0).to(device) for c in cls_list], dim=0)
        client_proto_sets.append(F.normalize(cp, dim=1))

    def step(x, y):
        feat, logits_s, logits_t, _ = dcr_forward(node, args, x)
        feat_n = F.normalize(feat, dim=1)

        sim_g = feat_n @ global_protos.t() / tau
        Lg = criterion(sim_g, y)

        Lp = 0.0
        for cp in client_proto_sets:
            sim_p = feat_n @ cp.t() / tau
            Lp = Lp + criterion(sim_p, y)
        Lp = Lp / max(1, len(client_proto_sets))

        kd, kd_w, do_kd = compute_dcr_kd(node, args, logits_s, logits_t)
        loss = Lg + Lp + kd_w * kd
        extras = {"Lg": Lg.item(), "Lp": float(Lp.item()), "kd": kd.item()}
        return loss, sim_g, extras, do_kd  # sim_g for accuracy

    avg_loss, avg_acc, ex = train_loop(node, args, step)
    if args.use_dcr:
        return avg_loss, ex["Lg"], ex["Lp"], ex["kd"], avg_acc
    return avg_loss, ex["Lg"], ex["Lp"], avg_acc


def train_fedtgp(node: Node, args):
    """
    FedTGP: CE + proto regularization + optional DCR KD.
    """
    global_protos = {}
    if hasattr(node, "glob_proto") and node.glob_proto is not None:
        for k, v in node.glob_proto.items():
            global_protos[k] = v.to(args.device)

    def step(x, y):
        feat, logits_s, logits_t, _ = dcr_forward(node, args, x)
        feat = check_and_fix_nan(feat, "fedtgp/features", replace_with_zero=True)

        ce = F.cross_entropy(logits_s, y)

        proto_reg = torch.tensor(0.0, device=args.device)
        if global_protos:
            proto_new = feat.clone().detach()
            for i, yy in enumerate(y):
                c = int(yy.item())
                if c in global_protos and global_protos[c] is not None:
                    proto_new[i, :] = global_protos[c].data
            proto_reg = F.mse_loss(proto_new, feat) * args.fedtgp_lambda

        kd, kd_w, do_kd = compute_dcr_kd(node, args, logits_s, logits_t)
        loss = ce + proto_reg + kd_w * kd
        extras = {"ce": ce.item(), "proto_reg": proto_reg.item(), "kd": kd.item()}
        return loss, logits_s, extras, do_kd

    avg_loss, avg_acc, ex = train_loop(node, args, step)
    if args.use_dcr:
        return avg_loss, ex["ce"], ex["proto_reg"], ex["kd"], avg_acc
    return avg_loss, ex["ce"], ex["proto_reg"], avg_acc


def train_fedplvm(node: Node, args):
    """
    FedPLVM: CE + lambda*(contra + corr) + optional DCR KD.
    """
    device = args.device
    sim_eps = 1e-6

    all_protos, all_labels = [], []
    for cls, proto_t in node.glob_proto.items():
        for p in proto_t:
            all_protos.append(p.view(-1))
            all_labels.append(cls)
    all_protos = torch.stack(all_protos, dim=0).to(device)
    all_labels = torch.tensor(all_labels, device=device)

    def step(x, y):
        feat, logits_s, logits_t, _ = dcr_forward(node, args, x)
        feat = check_and_fix_nan(feat, "fedplvm/features", replace_with_zero=True)

        ce = F.cross_entropy(logits_s, y)

        sim = F.cosine_similarity(feat.unsqueeze(1), all_protos.unsqueeze(0), dim=2).clamp(-1 + sim_eps, 1 - sim_eps)
        alpha = args.fedplvm_alpha
        s_alpha = torch.sign(sim) * sim.abs().pow(alpha)

        mask = (all_labels.unsqueeze(0) == y.unsqueeze(1))
        logits_proto = s_alpha / args.fedplvm_tau
        log_probs = logits_proto - torch.logsumexp(logits_proto, dim=1, keepdim=True)

        L_contra = -(log_probs * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        L_contra = L_contra.mean()

        C_y = mask.sum(dim=1).float()
        L_corr = torch.abs((s_alpha * mask).sum(dim=1) - C_y).mean()

        kd, kd_w, do_kd = compute_dcr_kd(node, args, logits_s, logits_t)
        loss = ce + args.fedplvm_lambda * (L_contra + L_corr) + kd_w * kd
        extras = {"ce": ce.item(), "contra": L_contra.item(), "corr": L_corr.item(), "kd": kd.item()}
        return loss, logits_s, extras, do_kd

    avg_loss, avg_acc, ex = train_loop(node, args, step)
    if args.use_dcr:
        return avg_loss, ex["ce"], ex["contra"], ex["corr"], ex["kd"], avg_acc
    return avg_loss, ex["ce"], ex["contra"], ex["corr"], avg_acc


def train_fpl(node: Node, args):
    """
    FPL: CE + w_contra*InfoNCE + w_mse*MSE + optional DCR KD.
    """
    def step(x, y):
        feat, logits_s, logits_t, _ = dcr_forward(node, args, x)
        ce = F.cross_entropy(logits_s, y)

        contra = calculate_infonce_loss(feat, y, node.glob_cluster_proto, args.contra_T) if node.glob_cluster_proto is not None else torch.tensor(0.0, device=args.device)

        if node.glob_unbiased_proto is not None:
            labels_in_batch = [int(lbl) for lbl in y]
            if all(lbl in node.glob_unbiased_proto for lbl in labels_in_batch):
                proto_vec = torch.stack([node.glob_unbiased_proto[lbl] for lbl in labels_in_batch])
                mse = F.mse_loss(feat, proto_vec)
            else:
                mse = torch.tensor(0.0, device=args.device)
        else:
            mse = torch.tensor(0.0, device=args.device)

        kd, kd_w, do_kd = compute_dcr_kd(node, args, logits_s, logits_t)
        loss = ce + args.fpl_contra_weight * contra + args.fpl_mse_weight * mse + kd_w * kd
        extras = {"ce": ce.item(), "contra": contra.item(), "mse": mse.item(), "kd": kd.item()}
        return loss, logits_s, extras, do_kd

    avg_loss, avg_acc, ex = train_loop(node, args, step)
    if args.use_dcr:
        return avg_loss, ex["ce"], ex["kd"], ex["contra"], ex["mse"], avg_acc
    return avg_loss, ex["ce"], None, ex["contra"], ex["mse"], avg_acc


def train_mpft(node, args):
    """
    MPFT: loss = CE + beta * KD + optional DCR KD.
    """
    teacher = node.teacher_model
    teacher.eval()

    node.model.train()
    ensure_dcr_state(node, args)

    T = args.mpft_kd_T
    beta = args.mpft_kd_beta

    total_loss = total_ce = total_kd = total_dcr = 0.0
    total_samples = total_correct = 0

    for x, y in node.train_loader:
        x = move_to_device(x, args.device)
        y = y.to(args.device)

        node.optimizer.zero_grad()

        with torch.no_grad():
            _, _, teacher_logits = teacher(x)

        feat, student_logits, logits_t, _ = dcr_forward(node, args, x)
        ce = F.cross_entropy(student_logits, y)

        S = F.log_softmax(student_logits / T, dim=1)
        Tgt = F.softmax(teacher_logits / T, dim=1)
        kd = F.kl_div(S, Tgt, reduction="batchmean") * (T * T)

        dcr_kd, dcr_w, do_kd = compute_dcr_kd(node, args, student_logits, logits_t)

        loss = ce + beta * kd + dcr_w * dcr_kd
        loss.backward()
        node.optimizer.step()

        if args.use_dcr and do_kd:
            ema_update_teacher(node.model, m=getattr(args, "ema_m", 0.999))
        if args.use_dcr:
            node._local_step += 1

        bs = y.size(0)
        total_samples += bs
        total_correct += (student_logits.argmax(dim=1) == y).sum().item()
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_kd += kd.item() * bs
        total_dcr += dcr_kd.item() * bs

    avg_loss = total_loss / max(1, total_samples)
    avg_ce = total_ce / max(1, total_samples)
    avg_kd = total_kd / max(1, total_samples)
    avg_dcr = total_dcr / max(1, total_samples)
    avg_acc = 100.0 * total_correct / max(1, total_samples)

    if args.use_dcr:
        return avg_loss, avg_ce, avg_kd, avg_dcr, avg_acc
    return avg_loss, avg_ce, avg_kd, avg_acc