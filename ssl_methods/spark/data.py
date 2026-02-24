"""SparK data pipeline — single augmented view per image.

SparK, like MAE, only needs one view per image: the masking provides the
self-supervised signal.  Augmentations are lighter than MoCo/DINO/BarlowTwins
— aggressive crops or flips are unnecessary because the reconstruction task
is already hard.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from data import collect_image_paths, _load_gray256


class GaussianNoise:
    """Add zero-mean Gaussian noise to a float tensor. Simulates X-ray quantum noise."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class SingleViewDataset(Dataset):
    """Returns one augmented view per image — sufficient for reconstruction-based SSL."""

    def __init__(
        self,
        image_paths: list[Path],
        transform: transforms.Compose,
        cache_in_ram: bool = False,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self._cache: list[np.ndarray] | None = None

        if cache_in_ram:
            print(f"  Caching {len(image_paths)} images as 256×256 grayscale arrays...", flush=True)
            self._cache = [_load_gray256(p) for p in image_paths]
            mb = sum(a.nbytes for a in self._cache) / 1024**2
            print(f"  Cache ready — {mb:.0f} MB in RAM.", flush=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache is not None:
            img = Image.fromarray(self._cache[idx], mode="L").convert("RGB")
        else:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def get_spark_transforms(config: dict) -> transforms.Compose:
    """Build the SparK augmentation pipeline for chest X-rays.

    Lighter than contrastive methods — the masking is the hard self-supervised
    task, so we avoid aggressive crops that would destroy anatomy.
    """
    aug = config["augmentations"]
    size = config["data"]["image_size"]

    pipeline = [
        transforms.RandomResizedCrop(size, scale=tuple(aug["random_crop_scale"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(aug.get("rotation_degrees", 10)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=aug["color_jitter_brightness"],
                contrast=aug["color_jitter_contrast"],
            )
        ], p=aug["color_jitter_prob"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if noise_std := aug.get("gaussian_noise_std", 0.0):
        pipeline.append(GaussianNoise(noise_std))
    return transforms.Compose(pipeline)


def build_spark_dataloader(config: dict) -> DataLoader:
    """Create a DataLoader for SparK pretraining."""
    transform = get_spark_transforms(config)
    datasets = []

    cache_in_ram = config["data"].get("cache_in_ram", False)
    for ds_cfg in config["data"]["datasets"]:
        paths = collect_image_paths(ds_cfg["name"], ds_cfg["root_dir"], ds_cfg.get("txt_file"))
        if paths:
            datasets.append(SingleViewDataset(paths, transform, cache_in_ram=cache_in_ram))
            print(f"  {ds_cfg['name']}: {len(paths)} images from {ds_cfg['root_dir']}")
        else:
            print(f"  {ds_cfg['name']}: WARNING — no images found in {ds_cfg['root_dir']}")

    if not datasets:
        raise RuntimeError("No images found across any configured datasets.")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    nw = config["data"]["num_workers"]
    return DataLoader(
        combined,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=False,  # Python 3.14 pin_memory bug
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )
