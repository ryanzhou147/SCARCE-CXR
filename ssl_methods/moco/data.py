"""MoCo v2 data pipeline: augmentation transforms and dataloader."""

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from data.dataset import UnlabeledChestXrayDataset, collect_image_paths


def get_moco_transforms(config: dict) -> transforms.Compose:
    """MoCo v2 augmentation pipeline for chest X-rays."""
    aug = config["augmentations"]
    size = config["data"]["image_size"]

    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=tuple(aug["random_crop_scale"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(aug.get("rotation_degrees", 10)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=aug["color_jitter_brightness"],
                contrast=aug["color_jitter_contrast"],
                saturation=aug["color_jitter_saturation"],
                hue=aug["color_jitter_hue"],
            )
        ], p=aug["color_jitter_prob"]),
        transforms.RandomGrayscale(p=aug["grayscale_prob"]),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=aug["gaussian_blur_kernel"]),
        ], p=aug["gaussian_blur_prob"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_moco_dataloader(config: dict) -> DataLoader:
    """Create a DataLoader for MoCo v2 pretraining from config."""
    transform = get_moco_transforms(config)
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
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4,
    )
