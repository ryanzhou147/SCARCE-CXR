"""MoCo v2 pretraining loop."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data import build_ssl_dataloader
from ssl_methods.moco.model import MoCo


def train_moco(config: dict) -> None:
    """Full MoCo v2 pretraining loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Building dataloader...")
    dataloader = build_ssl_dataloader(config)
    print(f"Total batches per epoch: {len(dataloader)}")

    moco_cfg = config["moco"]
    model = MoCo(
        encoder_name=moco_cfg["encoder"],
        dim=moco_cfg["dim"],
        K=moco_cfg["queue_size"],
        m=moco_cfg["momentum"],
        T=moco_cfg["temperature"],
    ).to(device)

    # MoCo uses SGD with momentum
    optimizer = torch.optim.SGD(
        model.encoder_q.parameters(),
        lr=config["training"]["lr"],
        momentum=0.9,
        weight_decay=config["training"]["weight_decay"],
    )

    warmup_epochs = config["training"]["warmup_epochs"]
    total_epochs = config["training"]["epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    best_loss = float("inf")
    wandb_run_id = None

    resume_path = output_dir / "latest.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        wandb_run_id = ckpt.get("wandb_run_id")

    wandb.init(
        project="cmvr-ssl",
        name=f"moco-{moco_cfg['encoder']}",
        id=wandb_run_id,
        resume="allow",
        config=config,
    )

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for x_q, x_k in pbar:
            x_q, x_k = x_q.to(device), x_k.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, labels = model(x_q, x_k)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        lr = optimizer.param_groups[0]["lr"]

        wandb.log({"loss/train": avg_loss, "lr": lr}, step=epoch)
        print(f"Epoch {epoch+1}/{total_epochs} — loss: {avg_loss:.4f}, lr: {lr:.6f}")

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": min(best_loss, avg_loss),
            "wandb_run_id": wandb.run.id,
            "config": config,
        }
        torch.save(ckpt_data, output_dir / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, output_dir / "best.pt")

        if (epoch + 1) % config["training"]["checkpoint_every"] == 0:
            torch.save(ckpt_data, output_dir / f"epoch_{epoch+1}.pt")

    wandb.finish()
    print(f"Training complete. Checkpoints saved to {output_dir}")
