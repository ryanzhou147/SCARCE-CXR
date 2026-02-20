"""CMVR — CLI entry point for SSL pretraining and downstream tasks."""

import argparse
import multiprocessing

import yaml

multiprocessing.set_start_method("fork", force=True)


def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply flat CLI overrides like --training.lr=1e-3
    for key, value in overrides.items():
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d[part]
        # Try to cast to the original type
        orig = d.get(parts[-1])
        if orig is not None:
            target_type = type(orig)
            if target_type is bool:
                value = value.lower() in ("true", "1", "yes")
            else:
                value = target_type(value)
        d[parts[-1]] = value

    return config


def parse_overrides(args: list[str]) -> dict:
    """Parse --key=value style overrides from remaining CLI args."""
    overrides = {}
    for arg in args:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            overrides[key] = value
    return overrides


def cmd_pretrain_moco(args: argparse.Namespace, remaining: list[str]) -> None:
    """Run MoCo v2 pretraining."""
    from ssl_methods.moco import train_moco

    overrides = parse_overrides(remaining)
    config = load_config(args.config, overrides)
    train_moco(config)


def cmd_pretrain_dino(args: argparse.Namespace, remaining: list[str]) -> None:
    """Run DINO pretraining."""
    from ssl_methods.dino import train_dino

    overrides = parse_overrides(remaining)
    config = load_config(args.config, overrides)
    train_dino(config)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cmvr", description="CMVR: Chest X-ray SSL Pretraining")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pretrain-moco
    p_moco = subparsers.add_parser("pretrain-moco", help="Run MoCo v2 self-supervised pretraining")
    p_moco.add_argument("--config", type=str, default="configs/moco.yaml", help="Path to config YAML")

    # pretrain-dino
    p_dino = subparsers.add_parser("pretrain-dino", help="Run DINO self-supervised pretraining")
    p_dino.add_argument("--config", type=str, default="configs/dino.yaml", help="Path to config YAML")

    args, remaining = parser.parse_known_args()

    if args.command == "pretrain-moco":
        cmd_pretrain_moco(args, remaining)
    elif args.command == "pretrain-dino":
        cmd_pretrain_dino(args, remaining)


if __name__ == "__main__":
    main()
