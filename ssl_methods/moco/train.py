"""MoCo v2 pretraining loop."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ssl_methods.moco.data import build_moco_dataloader
from ssl_methods.moco.model import MoCo


def train_moco(config: dict) -> None:
    """Full MoCo v2 pretraining loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Building dataloader...")
    dataloader = build_moco_dataloader(config)
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
    global_step = 0
    best_loss = float("inf")
    wandb_run_id = None

    resume_path = output_dir / "latest.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", start_epoch * len(dataloader))
        best_loss = checkpoint.get("best_loss", float("inf"))
        wandb_run_id = checkpoint.get("wandb_run_id")

    wandb.init(
        project="cmvr-ssl",
        name=config["training"]["run_name"],
        id=wandb_run_id,
        resume="allow",
        config=config,
    )

    epoch_pbar = tqdm(range(start_epoch, total_epochs), desc="Pretraining", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0

        batch_pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{total_epochs}", leave=False)
        for x_q, x_k in batch_pbar:
            x_q, x_k = x_q.to(device), x_k.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, labels = model(x_q, x_k)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            wandb.log({"loss/step": loss.item()}, step=global_step)
            global_step += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        lr = optimizer.param_groups[0]["lr"]

        epoch_pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{lr:.2e}")
        wandb.log({"loss/epoch": avg_loss, "lr": lr}, step=global_step)

        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": min(best_loss, avg_loss),
            "wandb_run_id": wandb.run.id,
            "config": config,
        }
        torch.save(checkpoint_data, output_dir / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint_data, output_dir / "best.pt")

        if (epoch + 1) % config["training"]["checkpoint_every"] == 0:
            torch.save(checkpoint_data, output_dir / f"epoch_{epoch+1}.pt")

    wandb.finish()
    print(f"Training complete. Checkpoints saved to {output_dir}")
