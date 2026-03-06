# Taming Noise-Induced Prototype Degradation for Privacy-Preserving Personalized Federated Fine-Tuning (CVPR 2026)

The implementation of paper Taming Noise-Induced Prototype Degradation for Privacy-Preserving Personalized Federated Fine-Tuning (CVPR 2026).

This repository targets federated prototype-based personalization (ProtoPFL) and implements VPDR as a client plug-in that can be integrated into existing ProtoPFL frameworks (e.g., FedProto), improving the privacy–utility trade-off over the classic equal-noise baseline (IGPP). The default configuration focuses on domain or label skew with ResNet / ViT backbones, but you can extend the framework to other datasets and model families.

## 1. Background & Method

- **Limitations of IGPP (equal noise)**  
  1. Feature dimensions contribute unevenly; applying the same noise to every axis over-perturbs the most discriminative ones and degrades prototype quality.  
  2. DP requires per-sample clipping, yet choosing a fixed \(\ell_2\) threshold is delicate: a large bound inflates the noise scale \(\Delta\), while a small bound severely distorts features.

- **VPDR plug-in (Variance-adaptive Prototype Perturbation + Distillation-guided Clipping Regularization)**  
  - **VPP** assigns noise adaptively across dimensions while keeping the same LDP guarantee, minimizing information loss.  
  - **DCR** introduces distillation-guided soft clipping during local personalization to stabilize per-sample norms.  
  - VPDR integrates with FedProto via flags such as `--noise_add vpp` and `--use_dcr`.

High-level workflow:
1. Each client applies VPP and DP perturbation to upload privatized prototypes.  
2. The server aggregates them into global prototypes.  
3. Clients run DCR-enhanced personalization on their private data.

## 2. Requirements

- Python 3.8+
- PyTorch ≥ 1.10 (GPU recommended)
- torchvision
- scikit-learn (FINCH / KMeans clustering)

## 3. Data & Model Preparation

- **Data**: the code loads CIFAR-10 from `data/`. Please download it manually (or rely on the script’s auto-download) and place it there.  
- **Pretrained models**: store the required ResNet/ViT checkpoints under `model/` (e.g., `vit-small/`, `vit-tiny/`). Any compatible weights are acceptable as long as the directory layout matches the code.

## 4. Example Run

```bash
python main.py \
  --dataset office_caltech10 \
  --node_num 4 \
  --T 20 \
  --E 2 \
  --model_type vit_small \
  --noise_add vpp \
  --use_dcr \
  --dcr_kd_weight 0.05 \
  --device cuda:0
```

Training logs, checkpoints, and metrics are stored under `logs/{exp_name}/...`, with summaries in `metrics.json`.

## 5. Directory Layout

```
├── main.py               # training entry point
├── options.py            # argument parser
├── client.py             # client-side prototype generation and local updates
├── server.py             # server-side prototype aggregation and updates
├── proto.py              # prototype construction, clustering, and perturbation
├── models.py             # model definitions and heterogeneous model factory
├── utils/
│   ├── init.py           # node, model, and optimizer initialization
│   ├── utils.py          # general utilities
│   ├── domain_skew.py    # domain-skew data loading
│   ├── label_skew.py     # label-skew data loading
│   ├── dp_utils.py       # differential privacy utilities
│   └── ...
└── README.md
```
 