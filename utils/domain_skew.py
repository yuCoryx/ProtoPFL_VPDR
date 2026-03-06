# domain_skew.py

import os
import random
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def partition_domain_skew_loaders(
    train_sets: List[Dataset],
    test_sets: List[Dataset],
    domains: List[str],
    keep_ratio: Dict[str, float],
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], int]:
    """
    Create one client per domain and subsample each domain's training set
    according to keep_ratio. Test sets are kept intact.
    """
    train_loaders, test_loaders = {}, {}

    for i, (train_ds, test_ds, domain) in enumerate(zip(train_sets, test_sets, domains)):
        total = len(train_ds)
        keep = max(1, int(total * keep_ratio[domain]))
        indices = random.sample(range(total), keep)

        train_loaders[i] = DataLoader(
            Subset(train_ds, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loaders[i] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    base_ds = train_sets[0].dataset if isinstance(train_sets[0], Subset) else train_sets[0]
    if hasattr(base_ds, "classes"):
        num_classes = len(base_ds.classes)
    elif hasattr(base_ds, "targets"):
        num_classes = int(np.max(np.array(base_ds.targets))) + 1
    else:
        raise ValueError(f"Cannot infer number of classes from dataset type {type(base_ds)}")

    return train_loaders, test_loaders, num_classes


def load_digits(domains: List[str], data_root: str, tfms: Dict[str, transforms.Compose]):
    """
    Load digit-domain datasets.
    """
    train_sets, test_sets = [], []

    for domain in domains:
        if domain == "mnist":
            train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfms["gray_train"])
            test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfms["gray_test"])
        elif domain == "usps":
            train_ds = datasets.USPS(os.path.join(data_root, "USPS"), train=True, download=True, transform=tfms["gray_train"])
            test_ds = datasets.USPS(os.path.join(data_root, "USPS"), train=False, download=True, transform=tfms["gray_test"])
        elif domain == "svhn":
            train_ds = datasets.SVHN(os.path.join(data_root, "SVHN"), "train", download=True, transform=tfms["rgb_train"])
            test_ds = datasets.SVHN(os.path.join(data_root, "SVHN"), "test", download=True, transform=tfms["rgb_test"])
        elif domain == "syn":
            train_ds = ImageFolder(os.path.join(data_root, "synthetic_digits", "imgs_train"), transform=tfms["rgb_train"])
            test_ds = ImageFolder(os.path.join(data_root, "synthetic_digits", "imgs_valid"), transform=tfms["rgb_test"])
        else:
            raise ValueError(f'Unknown digit domain: "{domain}"')

        train_sets.append(train_ds)
        test_sets.append(test_ds)

    return train_sets, test_sets


def load_pacs(domains: List[str], data_root: str, tfms: Dict[str, transforms.Compose], test_ratio: float = 0.2):
    """
    Load PACS domains.
    """
    train_sets, test_sets = [], []

    for domain in domains:
        base = os.path.join(data_root, "PACS", domain)
        train_dir = os.path.join(base, "train")
        test_dir = os.path.join(base, "test")

        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            train_ds = ImageFolder(train_dir, transform=tfms["rgb_train"])
            test_ds = ImageFolder(test_dir, transform=tfms["rgb_test"])
        else:
            full_ds = ImageFolder(base, transform=tfms["rgb_train"])
            n_total = len(full_ds)
            n_test = int(n_total * test_ratio)
            n_train = n_total - n_test
            train_ds, test_ds = random_split(full_ds, [n_train, n_test])

        train_sets.append(train_ds)
        test_sets.append(test_ds)

    return train_sets, test_sets


def load_office_caltech10(domains: List[str], data_root: str, tfms: Dict[str, transforms.Compose], test_ratio: float = 0.2):
    """
    Load Office-Caltech-10 domains. 
    """
    train_sets, test_sets = [], []

    for domain in domains:
        base = os.path.join(data_root, "Office-Caltech-10", domain)
        train_dir = os.path.join(base, "train")
        test_dir = os.path.join(base, "test")

        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            train_ds = ImageFolder(train_dir, transform=tfms["rgb_train"])
            test_ds = ImageFolder(test_dir, transform=tfms["rgb_test"])
        else:
            full_ds = ImageFolder(base, transform=tfms["rgb_train"])
            n_total = len(full_ds)
            n_test = int(n_total * test_ratio)
            n_train = n_total - n_test
            train_ds, test_ds = random_split(full_ds, [n_train, n_test])

        train_sets.append(train_ds)
        test_sets.append(test_ds)

    return train_sets, test_sets


def partition_domain_label_skew_loaders(
    train_sets: List[Dataset],
    test_sets: List[Dataset],
    domains: List[str],
    keep_ratio: Dict[str, float],
    alpha: float,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], int]:
    """
    Apply label skew within each domain using a Dirichlet distribution,
    then subsample the resulting training set by keep_ratio.
    """
    train_loaders, test_loaders = {}, {}

    base_ds = train_sets[0].dataset if isinstance(train_sets[0], Subset) else train_sets[0]
    if hasattr(base_ds, "classes"):
        num_classes = len(base_ds.classes)
    else:
        num_classes = int(np.max(np.array(base_ds.targets))) + 1

    for i, (train_ds, test_ds, domain) in enumerate(zip(train_sets, test_sets, domains)):
        n_train = len(train_ds)
        ys = np.array([train_ds[j][1] for j in range(n_train)])
        idxs_by_class = [np.where(ys == c)[0].tolist() for c in range(num_classes)]

        props = np.random.dirichlet(alpha * np.ones(num_classes))
        counts = np.random.multinomial(n_train, props)

        sampled = []
        for c, cnt in enumerate(counts):
            pool = idxs_by_class[c]
            random.shuffle(pool)
            sampled.extend(pool[:cnt])

        random.shuffle(sampled)
        keep = max(1, int(len(sampled) * keep_ratio[domain]))
        sampled = random.sample(sampled, keep)

        train_loaders[i] = DataLoader(
            Subset(train_ds, sampled),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loaders[i] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loaders, test_loaders, num_classes


def get_federated_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], int]:
    """
    Main entry point for domain-skew federated data loading.
    Returns:
        train_loaders, test_loaders, num_classes
    """
    tfms = {
        "gray_train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "gray_test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "rgb_train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "rgb_test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
    }

    if dataset == "digits":
        domains = ["mnist", "usps", "svhn", "syn"]
        keep_ratio = {d: 0.1 for d in domains}
        train_sets, test_sets = load_digits(domains, data_root, tfms)

    elif dataset == "pacs":
        domains = ["photo", "art_painting", "cartoon", "sketch"]
        keep_ratio = {d: 0.3 for d in domains}
        train_sets, test_sets = load_pacs(domains, data_root, tfms)

    elif dataset == "office_caltech10":
        domains = ["amazon", "caltech", "dslr", "webcam"]
        keep_ratio = {d: 0.3 for d in domains}
        train_sets, test_sets = load_office_caltech10(domains, data_root, tfms)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return partition_domain_skew_loaders(
        train_sets=train_sets,
        test_sets=test_sets,
        domains=domains,
        keep_ratio=keep_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
    )