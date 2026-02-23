"""DINO multi-crop data pipeline.

Produces ``n_global`` large-crop views (used by teacher + student) and
``n_local`` small-crop views (used by student only).  A custom collate
function stacks each view position across the batch so the training loop
receives a plain ``list[Tensor]``.
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


class MultiCropTransform:
    """Applies two augmentation pipelines to yield multiple views per image.

    The first ``n_global`` views come from ``global_transform`` (large crops).
    The remaining ``n_local`` views come from ``local_transform`` (small crops).
    """

    def __init__(
        self,
        global_transform: transforms.Compose,
        local_transform: transforms.Compose,
        n_global: int = 2,
        n_local: int = 6,
    ):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, img) -> list[torch.Tensor]:
        views = [self.global_transform(img) for _ in range(self.n_global)]
        views += [self.local_transform(img) for _ in range(self.n_local)]
        return views


class MultiCropDataset(Dataset):
    """Returns a list of augmented views per image."""

    def __init__(
        self,
        image_paths: list[Path],
        transform: MultiCropTransform,
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

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        if self._cache is not None:
            img = Image.fromarray(self._cache[idx], mode="L").convert("RGB")
        else:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def multicrop_collate(batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Stack each view position across the batch into a ``(B, C, H, W)`` tensor."""
    n_views = len(batch[0])
    return [torch.stack([item[i] for item in batch]) for i in range(n_views)]


def get_dino_transforms(config: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """Build ``(global_transform, local_transform)`` from config."""
    aug = config["augmentations"]
    dino_cfg = config["dino"]
    global_size = config["data"]["image_size"]
    local_size = dino_cfg.get("local_crop_size", 96)
    global_scale = tuple(dino_cfg.get("global_crop_scale", [0.4, 1.0]))
    local_scale = tuple(dino_cfg.get("local_crop_scale", [0.05, 0.4]))

    shared = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(aug.get("rotation_degrees", 10)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=aug["color_jitter_brightness"],
                contrast=aug["color_jitter_contrast"],
            )
        ], p=aug["color_jitter_prob"]),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=aug["gaussian_blur_kernel"]),
        ], p=aug["gaussian_blur_prob"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if noise_std := aug.get("gaussian_noise_std", 0.0):
        shared.append(GaussianNoise(noise_std))

    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(global_size, scale=global_scale),
        *shared,
    ])
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(local_size, scale=local_scale),
        *shared,
    ])
    return global_transform, local_transform


def build_dino_dataloader(config: dict) -> DataLoader:
    """Create a multi-crop DataLoader for DINO pretraining."""
    dino_cfg = config["dino"]
    n_global = dino_cfg.get("n_global_crops", 2)
    n_local = dino_cfg.get("n_local_crops", 6)

    global_transform, local_transform = get_dino_transforms(config)
    multicrop = MultiCropTransform(global_transform, local_transform, n_global, n_local)

    cache_in_ram = config["data"].get("cache_in_ram", False)
    datasets = []
    for ds_cfg in config["data"]["datasets"]:
        paths = collect_image_paths(ds_cfg["name"], ds_cfg["root_dir"], ds_cfg.get("txt_file"))
        if paths:
            datasets.append(MultiCropDataset(paths, multicrop, cache_in_ram=cache_in_ram))
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
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        collate_fn=multicrop_collate,
    )
