"""BarlowTwins data pipeline.

Identical structure to MoCo: returns two independently augmented views of
each image.  The augmentation pipeline is tuned for chest X-rays (no
saturation/hue jitter, which are no-ops on grayscale; Gaussian noise added
to simulate X-ray quantum noise).
"""

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from data.dataset import UnlabeledChestXrayDataset, collect_image_paths


class GaussianNoise:
    """Add zero-mean Gaussian noise to a float tensor. Simulates X-ray quantum noise."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


def get_barlow_transforms(config: dict) -> transforms.Compose:
    """Build the augmentation pipeline for BarlowTwins chest X-ray pretraining."""
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
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=aug["gaussian_blur_kernel"]),
        ], p=aug["gaussian_blur_prob"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if noise_std := aug.get("gaussian_noise_std", 0.0):
        pipeline.append(GaussianNoise(noise_std))
    return transforms.Compose(pipeline)


def build_barlow_dataloader(config: dict) -> DataLoader:
    """Create a DataLoader for BarlowTwins pretraining."""
    transform = get_barlow_transforms(config)
    datasets = []

    cache_in_ram = config["data"].get("cache_in_ram", False)
    for ds_cfg in config["data"]["datasets"]:
        paths = collect_image_paths(ds_cfg["name"], ds_cfg["root_dir"], ds_cfg.get("txt_file"))
        if paths:
            datasets.append(UnlabeledChestXrayDataset(paths, transform, cache_in_ram=cache_in_ram))
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
        prefetch_factor=(2 if nw > 0 else None),  # cache_in_ram=True: no I/O bottleneck, keep shm low
    )
