import importlib.util
import os

import torch

from transferlearn.finetune import ViTFinetuner


ATTACK_ENTRYPOINTS = {
    "PGD": ("attack-pgd.py", "run_pgd_attack"),
}


def _backbone_name(args):
    backbone = getattr(args, "target_model", None)
    if backbone in (None, "none"):
        backbone = "vitb16"
    return backbone


def _build_finetuner(args, device):
    return ViTFinetuner(
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        tune_mode=args.ftmode,
        backbone=_backbone_name(args),
    )


def _load_state_dict_model(args, device, checkpoint_path):
    finetuner = _build_finetuner(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = finetuner.model.load_state_dict(state_dict, strict=False)
    if _backbone_name(args) != "vitb16" and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint does not match backbone '{_backbone_name(args)}': "
            f"missing={missing[:5]} unexpected={unexpected[:5]}")
    finetuner.model.eval()
    finetuner.model.to(device)
    return finetuner.model


def _load_finetuned_model(args, device):
    ckpt_path = getattr(args, "ckpt_path", None)
    if not ckpt_path:
        raise ValueError("Please set --ckpt_path to the fine-tuned model state dict (.pth).")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")
    print(f"Loading finetuned model state dict from {ckpt_path}")
    return _load_state_dict_model(args, device, ckpt_path)


def _load_attack_runner(attack_name, root_dir):
    if attack_name not in ATTACK_ENTRYPOINTS:
        valid = ", ".join(sorted(ATTACK_ENTRYPOINTS.keys()))
        raise ValueError(f"Unsupported attack '{attack_name}'. Valid: {valid}")

    attack_file_name, entrypoint = ATTACK_ENTRYPOINTS[attack_name]
    attack_file = os.path.join(root_dir, "attack", attack_file_name)
    module_name = f"attack_{attack_name.lower()}"
    spec = importlib.util.spec_from_file_location(module_name, attack_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load attack file: {attack_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_attack = getattr(module, entrypoint, None)
    if run_attack is None:
        raise AttributeError(f"{entrypoint} not found in {attack_file}")
    return run_attack


def run_attack_target_model(args):
    if not getattr(args, "dataset", None):
        args.dataset = getattr(args, "target_dataset", None)
    if not getattr(args, "ftmode", None):
        args.ftmode = getattr(args, "ft_mode", None)
    if not getattr(args, "attack", None):
        args.attack = getattr(args, "attack_mode", None)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.attack == "none":
        raise ValueError("Please set --attack_mode to PGD")

    print(args)
    args.target_model_name = f"vit_finetuned_{args.dataset}_{args.ftmode}"
    args.root_path = root_dir
    args.backbone = _backbone_name(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu = getattr(args, "gpu", "0")
        try:
            gpu_id = int(str(gpu).split(",")[0])
        except (ValueError, TypeError):
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        torch.cuda.init()
    target_model = _load_finetuned_model(args, device)
    run_attack = _load_attack_runner(args.attack, root_dir)
    run_attack(args, target_model, device)
