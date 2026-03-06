# label_skew.py

from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_dirichlet_partitions(dataset, num_clients, alpha):
    """Partition dataset indices across clients with a Dirichlet label distribution."""
    y = np.array(dataset.targets)
    num_classes = int(y.max()) + 1
    idx_by_class = [np.where(y == c)[0] for c in range(num_classes)]

    client_map = {i: [] for i in range(num_clients)}
    for c_idxs in idx_by_class:
        np.random.shuffle(c_idxs)
        props = np.random.dirichlet(alpha * np.ones(num_clients))
        counts = (props * len(c_idxs)).astype(int)
        counts[-1] = len(c_idxs) - counts[:-1].sum()

        start = 0
        for cid, cnt in enumerate(counts):
            client_map[cid].extend(c_idxs[start:start + cnt].tolist())
            start += cnt

    return client_map


class LabelSkewDataModule:
    """
    Label-skew partitioning for CIFAR datasets.
    Each client gets a subset of the training set, while all clients share the same global test set.
    """
    def __init__(self, dataset_name: str, data_root: str, num_clients: int, alpha: float, batch_size: int, num_workers: int = 4):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._setup()

    def _setup(self):
        mean_std = {
            "cifar10": ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            "cifar100": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        }
        dataset_map = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
        }

        if self.dataset_name not in dataset_map:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        mean, std = mean_std[self.dataset_name]
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        Dcls = dataset_map[self.dataset_name]
        self.train_set = Dcls(self.data_root, train=True, download=True, transform=tf)
        self.test_set = Dcls(self.data_root, train=False, download=True, transform=tf)

        if self.alpha == 0:
            n = len(self.train_set)
            per = n // self.num_clients
            self.client_map = {i: list(range(i * per, (i + 1) * per)) for i in range(self.num_clients)}
        else:
            self.client_map = build_dirichlet_partitions(self.train_set, self.num_clients, self.alpha)

    def get_loaders(self) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], int]:
        train_loaders = {}
        for cid, idxs in self.client_map.items():
            train_loaders[cid] = DataLoader(
                Subset(self.train_set, idxs),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        global_test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loaders = {cid: global_test_loader for cid in range(self.num_clients)}
        test_loaders[-1] = global_test_loader

        num_classes = int(np.max(np.array(self.train_set.targets))) + 1
        return train_loaders, test_loaders, num_classes


def get_federated_loaders(
    dataset: str,
    data_root: str,
    num_clients: int,
    batch_size: int,
    alpha: float,
    num_workers: int = 4,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], int]:
    """Main entry point for CIFAR label-skew federated data loading."""
    dm = LabelSkewDataModule(
        dataset_name=dataset,
        data_root=data_root,
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return dm.get_loaders()