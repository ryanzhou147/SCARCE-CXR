"""DINO pretraining loop."""

import math
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ssl_methods.dino.data import build_dino_dataloader
from ssl_methods.dino.loss import DINOLoss
from ssl_methods.dino.model import DINO


def train_dino(config: dict) -> None:
    """Full DINO pretraining loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Building dataloader...")
    dataloader = build_dino_dataloader(config)
    print(f"Total batches per epoch: {len(dataloader)}")

    dino_cfg = config["dino"]
    n_global = dino_cfg["n_global_crops"]
    out_dim = dino_cfg["out_dim"]

    model = DINO(encoder_name=dino_cfg["encoder"], out_dim=out_dim).to(device)

    criterion = DINOLoss(
        out_dim=out_dim,
        n_global_crops=n_global,
        student_temp=dino_cfg["student_temp"],
        teacher_temp=dino_cfg["teacher_temp"],
        center_momentum=dino_cfg["center_momentum"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=config["training"]["lr"],
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

    # Per-step cosine schedule: teacher EMA momentum ramps from start → end.
    total_steps = total_epochs * len(dataloader)
    momentum_schedule = torch.linspace(
        dino_cfg["teacher_momentum_start"],
        dino_cfg["teacher_momentum_end"],
        total_steps,
    )

    # Teacher temperature warmup: linearly decay from warmup_start → teacher_temp
    # over the first teacher_temp_warmup_epochs epochs, then hold.
    teacher_temp_final = dino_cfg["teacher_temp"]
    teacher_temp_start = dino_cfg.get("teacher_temp_warmup_start", teacher_temp_final)
    teacher_temp_warmup_epochs = dino_cfg.get("teacher_temp_warmup_epochs", 0)

    def get_teacher_temp(epoch: int) -> float:
        if teacher_temp_warmup_epochs <= 0 or epoch >= teacher_temp_warmup_epochs:
            return teacher_temp_final
        frac = epoch / teacher_temp_warmup_epochs
        return teacher_temp_start + frac * (teacher_temp_final - teacher_temp_start)

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
        criterion.load_state_dict(checkpoint["criterion"])
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
        criterion.teacher_temp = get_teacher_temp(epoch)
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0

        batch_pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{total_epochs}", leave=False)
        for views in batch_pbar:
            views = [v.to(device) for v in views]

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                student_out, teacher_out = model(views, n_global)
                loss = criterion(student_out, teacher_out)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            step_idx = min(global_step, total_steps - 1)
            teacher_momentum = float(momentum_schedule[step_idx])
            model.update_teacher(teacher_momentum)
            global_step += 1

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            wandb.log({"loss/step": loss.item()}, step=global_step)
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_grad_norm = total_grad_norm / len(dataloader)
        lr = optimizer.param_groups[0]["lr"]
        epoch_start_step = epoch * len(dataloader)
        epoch_momentum = float(momentum_schedule[min(epoch_start_step, total_steps - 1)])

        epoch_pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{lr:.2e}", t_temp=f"{criterion.teacher_temp:.4f}")
        wandb.log({
            "loss/epoch": avg_loss,
            "lr": lr,
            "teacher_momentum": epoch_momentum,
            "teacher_temp": criterion.teacher_temp,
            "grad_norm": avg_grad_norm,
        }, step=global_step)

        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "criterion": criterion.state_dict(),
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
