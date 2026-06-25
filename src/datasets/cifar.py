from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


CIFAR_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "filename": "cifar-10-python.tar.gz",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "folder": "cifar-10-batches-py",
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "filename": "cifar-100-python.tar.gz",
        "md5": "eb9058c3a382ffc7106e4002c42a8d85",
        "folder": "cifar-100-python",
    },
}


class Compose:
    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


class Normalize:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p:
            return torch.flip(image, dims=(2,))
        return image


class RandomCrop:
    def __init__(self, size: int, padding: int = 0) -> None:
        self.size = size
        self.padding = padding

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            image = torch.nn.functional.pad(image, (self.padding,) * 4)
        _, height, width = image.shape
        if height == self.size and width == self.size:
            return image
        top = torch.randint(0, height - self.size + 1, ()).item()
        left = torch.randint(0, width - self.size + 1, ()).item()
        return image[:, top : top + self.size, left : left + self.size]


class CIFARDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        name: str,
        train: bool,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        normalized_name = name.lower().replace("-", "")
        if normalized_name not in CIFAR_STATS:
            raise ValueError(f"Unsupported dataset '{name}'. Use cifar10 or cifar100.")
        self.name = normalized_name
        self.root = Path(root)
        self.train = train
        self.transform = transform

        if download:
            download_cifar(self.root, self.name)

        self.data, self.targets = self._load()

    def _load_pickle(self, path: Path) -> dict:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1")

    def _load(self) -> tuple[np.ndarray, list[int]]:
        info = CIFAR_STATS[self.name]
        dataset_dir = self.root / info["folder"]
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Missing {dataset_dir}. Set download: true in config or place the extracted CIFAR files there."
            )

        arrays: list[np.ndarray] = []
        targets: list[int] = []

        if self.name == "cifar10":
            files = [f"data_batch_{idx}" for idx in range(1, 6)] if self.train else ["test_batch"]
            label_key = "labels"
        else:
            files = ["train"] if self.train else ["test"]
            label_key = "fine_labels"

        for filename in files:
            entry = self._load_pickle(dataset_dir / filename)
            arrays.append(entry["data"])
            targets.extend(entry[label_key])

        data = np.concatenate(arrays, axis=0).reshape(-1, 3, 32, 32)
        return data, targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.tensor(self.data[index], dtype=torch.float32).div_(255.0)
        label = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cifar(root: Path, name: str) -> None:
    info = CIFAR_STATS[name]
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / info["filename"]
    dataset_dir = root / info["folder"]
    if dataset_dir.exists():
        return

    if not archive_path.exists():
        print(f"Downloading {name} to {archive_path}")
        urllib.request.urlretrieve(info["url"], archive_path)

    expected_md5 = info["md5"]
    if md5(archive_path) != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {archive_path}. Delete it and retry.")

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(root)


def build_cifar_transforms(name: str, train: bool) -> Compose:
    info = CIFAR_STATS[name.lower().replace("-", "")]
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []
    if train:
        transforms.extend([RandomCrop(32, padding=4), RandomHorizontalFlip()])
    transforms.append(Normalize(info["mean"], info["std"]))
    return Compose(transforms)


def build_cifar_loaders(config: dict) -> tuple[DataLoader, DataLoader, int]:
    dataset_name = config["dataset"]["name"].lower().replace("-", "")
    root = config["dataset"].get("root", "data")
    download = bool(config["dataset"].get("download", False))
    batch_size = int(config["training"].get("batch_size", 128))
    num_workers = int(config["training"].get("num_workers", 0))
    val_split = int(config["dataset"].get("val_split", 0))
    seed = int(config.get("seed", 42))

    train_dataset = CIFARDataset(
        root=root,
        name=dataset_name,
        train=True,
        transform=build_cifar_transforms(dataset_name, train=True),
        download=download,
    )

    eval_train_dataset = train_dataset
    if val_split > 0:
        train_size = len(train_dataset) - val_split
        generator = torch.Generator().manual_seed(seed)
        eval_train_dataset, _ = random_split(train_dataset, [train_size, val_split], generator=generator)

    test_dataset = CIFARDataset(
        root=root,
        name=dataset_name,
        train=False,
        transform=build_cifar_transforms(dataset_name, train=False),
        download=download,
    )

    train_loader = DataLoader(
        eval_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, int(CIFAR_STATS[dataset_name]["num_classes"])

