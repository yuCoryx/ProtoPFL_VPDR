import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Reproducibility & experiment
    parser.add_argument('--random_seed', type=int, default=10, help='Random seed for the whole experiment.')
    parser.add_argument('--exp_name', type=str, default='logs', help='Experiment name (used in output path).')

    # Data & environment
    parser.add_argument('--data_skew', type=str, default='domain', choices=['label', 'domain'], help='Client data partition type: "label" (Dirichlet label skew) or "domain" (one domain per client).')
    parser.add_argument('--device', type=str, default='cuda:3', help='Training device, e.g., "cuda:0", "cuda:3", or "cpu".')
    parser.add_argument('--data_root', type=str, default='data/', help='Dataset root directory.')
    parser.add_argument('--dataset', type=str, default='pacs', help='Dataset name, e.g., {cifar10, digits, pacs, office_caltech10}.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')

    # Model
    parser.add_argument('--model_type', type=str, default='vit_tiny', help='Local model type, e.g., {resnet18, vit_tiny, vit_small, vit_base}.')
    parser.add_argument('--unfreeze_layers', type=str, default='ad_cla', help='Layer names/patterns to unfreeze (implementation-dependent).')

    # Heterogeneous model settings
    parser.add_argument('--enable_heterogeneous', action='store_true', help='Enable heterogeneous client models.')
    parser.add_argument('--model_family', type=str, default='HtFE4', choices=['HtFE2', 'HtFE4', 'HtFE6', 'HtC4', 'HtFE4-HtC2', 'ResNet4', 'ViT4'], help='Heterogeneous model family configuration.')
    parser.add_argument('--feature_dim', type=int, default=512, help='Unified feature dimension: all heterogeneous models output the same feature dimension.')

    # Federated learning setup
    parser.add_argument('--dirichlet_alpha', type=float, default=1, help='Dirichlet alpha for label skew. 0 means IID (depending on your loader implementation).')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--method', type=str, default='fedproto', help='Federated method: {fedproto, fedplvm, fpl, fedpcl, mpft, fedtgp, fedvpd}.')
    parser.add_argument('--node_num', type=int, default=20, help='Number of clients.')
    parser.add_argument('--T', type=int, default=20, help='Number of communication rounds.')
    parser.add_argument('--E', type=int, default=2, help='Number of local epochs per round.')

    # Prototype clustering / generation
    parser.add_argument('--cluster_method', type=str, default='finch', help='Prototype clustering method, e.g., {mean, finch, kmeans}.')
    parser.add_argument('--cluster_rate', type=float, default=0.1, help='Clustering rate / ratio (meaning depends on your implementation).')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer type: {sgd, adam, adamw}.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (only used when optimizer=sgd).')

    # Differential privacy (DP)
    parser.add_argument('--privacy', type=str, default='dp', help='Privacy mode, e.g., "dp" or "nodp" (depending on your code path).')
    parser.add_argument('--noise_add', type=str, default='equal', choices=['equal', 'vpp'], help='Prototype noise mechanism: "equal" (standard) or "vpp" (VPP partition + DP top-k).')
    parser.add_argument('--vpp_rho', type=float, default=0.3, help='Discriminative subspace ratio rho (< 0.5).')
    parser.add_argument('--vpp_topk_eps_ratio', type=float, default=0.1, help='Privacy budget ratio for one-shot DP top-k selection: eps_topk = eps_total * ratio.')
    parser.add_argument('--vpp_topk_delta', type=float, default=0, help='Delta for one-shot DP top-k selection.')
    parser.add_argument('--clip_proto_norm', type=float, default=5.0, help='Per-sample L2 clipping norm (for features/prototypes, depending on your pipeline).')
    parser.add_argument('--epsilon', type=float, default=10.0, help='DP epsilon.')
    parser.add_argument('--delta', type=float, default=1e-5, help='DP delta.')

    # DCR plugin (distillation-guided clipping regularization)
    parser.add_argument('--use_dcr', action='store_true', help='Enable CTR/DCR plugin (distillation-guided clipping regularization).')
    parser.add_argument('--softclip_tau', type=float, default=0.05, help='Soft clipping temperature/tau for continuous clipping approximation.')
    parser.add_argument('--dcr_kd_weight', type=float, default=0.05, help='CTR/DCR distillation loss weight (teacher path -> student path).')
    parser.add_argument('--dcr_kd_T', type=float, default=4.0, help='CTR/DCR distillation temperature (teacher path -> student path).')
    parser.add_argument('--ema_m', type=float, default=0.999, help='EMA momentum (0.999 recommended; larger = smoother).')
    parser.add_argument('--kd_warmup_steps', type=int, default=5, help='KD warm-up steps: disable KD/EMA for the first N local steps.')

    # ProtoFLs-specific hyperparameters
    parser.add_argument('--fedproto_mse_weight', type=float, default=0.1, help='FedProto MSE prototype matching loss weight.')
    parser.add_argument('--fedpcl_tau', type=float, default=0.07, help='FedPCL contrastive temperature.')
    parser.add_argument('--fedplvm_alpha', type=float, default=0.25, help='FedPLVM sparsity parameter alpha.')
    parser.add_argument('--fedplvm_tau', type=float, default=0.07, help='FedPLVM temperature.')
    parser.add_argument('--fedplvm_lambda', type=float, default=0.5, help='FedPLVM loss scaling factor lambda.')
    parser.add_argument('--fpl_contra_weight', type=float, default=0.5, help='FPL contrastive loss weight.')
    parser.add_argument('--fpl_mse_weight', type=float, default=0.5, help='FPL MSE prototype matching loss weight.')
    parser.add_argument('--fedtgp_lambda', type=float, default=1.0, help='FedTGP prototype regularization weight.')
    parser.add_argument('--fedtgp_margin_threshold', type=float, default=100.0, help='FedTGP margin learning threshold.')
    parser.add_argument('--fedtgp_server_epochs', type=int, default=100, help='FedTGP server-side training epochs.')
    parser.add_argument('--mpft_kd_T', type=float, default=4.0, help='MPFT distillation temperature.')
    parser.add_argument('--mpft_kd_beta', type=float, default=0.3, help='MPFT distillation coefficient beta.')
    parser.add_argument('--mpft_sepoch', type=int, default=20, help='MPFT server fine-tuning epochs (server epochs).')

    # Privacy attack evaluation
    parser.add_argument('--eval_hijack', action='store_true', help='Enable hijack attack evaluation (run every round).')
    parser.add_argument('--hijack_per_class', type=int, default=1, help='Number of prototypes attacked per class.')
    parser.add_argument('--hijack_max_classes', type=int, default=3, help='Maximum number of classes to attack.')
    parser.add_argument('--hijack_steps', type=int, default=100000, help='Maximum optimization steps for the attack.')
    parser.add_argument('--hijack_lr', type=float, default=0.01, help='Attack optimizer learning rate (0.01 recommended).')
    parser.add_argument('--hijack_tv', type=float, default=1e-3, help='Total Variation regularization weight.')
    parser.add_argument('--hijack_l2', type=float, default=0.0, help='L2 regularization weight.')
    parser.add_argument('--hijack_aug', action='store_true', help='Enable augmentation consistency to reduce overfitting during attack.')
    parser.add_argument('--hijack_early_stop_patience', type=int, default=500, help='Hijack early stopping: stop if no improvement for N steps.')
    parser.add_argument('--hijack_early_stop_threshold', type=float, default=1e-6, help='Hijack early stopping: improvement smaller than this is treated as no improvement.')
    parser.add_argument('--hijack_min_steps', type=int, default=100, help='Minimum steps before early stopping can trigger.')
    parser.add_argument('--hijack_batch_size', type=int, default=16, help='Attack batch size (1=sequential; >1=batch optimization).')

    parser.add_argument('--eval_mia', action='store_true', help='Enable MIA evaluation (run every round).')
    parser.add_argument('--mia_comprehensive', action='store_true', help='Compute comprehensive metrics (AUC, TPR@FPR, advantage, etc.).')
    parser.add_argument('--mia_target_fprs', type=float, nargs='+', default=[0.01, 0.001], help='Target false positive rates for reporting TPR@FPR (e.g., 0.01 0.001).')
    parser.add_argument('--mia_max_per_class', type=int, default=800, help='Max number of train/test features collected per class.')

    args = parser.parse_args()
    return args