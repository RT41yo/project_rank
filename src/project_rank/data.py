from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SyntheticCIFAR(Dataset[tuple[torch.Tensor, int]]):
    """Small deterministic dataset used only for smoke tests."""

    def __init__(self, size: int, num_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 3, 32, 32, generator=generator)
        self.targets = torch.randint(
            low=0,
            high=num_classes,
            size=(size,),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.targets[index])


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_loaders(config: Mapping[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    dataset_name = str(config.get("dataset", "cifar10")).lower()
    seed = int(config.get("seed", 1))
    batch_size = int(config.get("batch_size", 128))
    workers = int(config.get("num_workers", 4))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
    num_classes = int(config.get("num_classes", 10))

    if dataset_name == "synthetic":
        train_set = SyntheticCIFAR(
            int(config.get("synthetic_train_size", 64)), num_classes, seed
        )
        val_set = SyntheticCIFAR(
            int(config.get("synthetic_val_size", 32)), num_classes, seed + 1
        )
    elif dataset_name == "cifar10":
        from torchvision import datasets, transforms

        data_dir = str(config.get("data_dir", "data"))
        val_size = int(config.get("val_size", 5000))
        if not 0 < val_size < 50_000:
            raise ValueError("val_size must be between 1 and 49,999")

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        train_source = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        val_source = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=eval_transform
        )
        indices = torch.randperm(50_000, generator=torch.Generator().manual_seed(seed))
        val_indices = indices[:val_size].tolist()
        train_indices = indices[val_size:].tolist()
        train_set = Subset(train_source, train_indices)
        val_set = Subset(val_source, val_indices)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = _loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        seed=seed,
        pin_memory=pin_memory,
    )
    val_loader = _loader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        seed=seed,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_test_loader(config: Mapping[str, Any]) -> DataLoader[Any]:
    """Build a deterministic loader for the held-out test split.

    CIFAR-10 test images are never used by ``build_loaders`` and therefore do
    not influence training, architecture selection, or checkpoint selection.
    """
    dataset_name = str(config.get("dataset", "cifar10")).lower()
    seed = int(config.get("seed", 1))
    batch_size = int(config.get("batch_size", 128))
    workers = int(config.get("num_workers", 4))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
    num_classes = int(config.get("num_classes", 10))

    if dataset_name == "synthetic":
        test_set: Dataset[Any] = SyntheticCIFAR(
            int(config.get("synthetic_test_size", 32)), num_classes, seed + 2
        )
    elif dataset_name == "cifar10":
        from torchvision import datasets, transforms

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        test_set = datasets.CIFAR10(
            str(config.get("data_dir", "data")),
            train=False,
            download=True,
            transform=eval_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return _loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        seed=seed,
        pin_memory=pin_memory,
    )

