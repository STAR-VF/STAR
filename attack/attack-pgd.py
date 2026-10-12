import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


VIT_NORM_MEAN = (0.5, 0.5, 0.5)
VIT_NORM_STD = (0.5, 0.5, 0.5)
IMAGENET_NORM_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORM_STD = (0.229, 0.224, 0.225)


def _norm_stats(args):
    if getattr(args, "backbone", "vitb16") == "dinov2b14":
        return IMAGENET_NORM_MEAN, IMAGENET_NORM_STD
    return VIT_NORM_MEAN, VIT_NORM_STD


def clear_and_create_images_dir(images_dir):
    """Clear the images directory and recreate it."""
    if os.path.exists(images_dir):
        for name in os.listdir(images_dir):
            path = os.path.join(images_dir, name)
            if os.path.isdir(path):
                for sub in os.listdir(path):
                    os.remove(os.path.join(path, sub))
                os.rmdir(path)
            else:
                os.remove(path)
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def _save_images(inps, filenames, output_dir):
    for i, filename in enumerate(filenames):
        save_path = os.path.join(output_dir, filename)
        image = inps[i].permute(1, 2, 0)
        image = torch.clamp(image, min=0.0, max=1.0)
        image = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))
        image.save(save_path)


def run_pgd_attack(args, model, device):
    """Run PGD Linf attack on the finetuned model."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    base_dir = os.path.abspath(args.save_path)
    images_dir = clear_and_create_images_dir(os.path.join(base_dir, "images"))

    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    norm_mean, norm_std = _norm_stats(args)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    data_path = os.path.join(project_root, "data")

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=transform,
        )
    elif args.dataset == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_path,
            train=False,
            download=True,
            transform=transform,
        )
    elif args.dataset == "oxford":
        dataset = datasets.OxfordIIITPet(
            root=data_path,
            split="test",
            target_types="category",
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset for PGD: {args.dataset}")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Attack: PGD, Model: vit_finetuned_{args.dataset}_{args.ftmode}")

    model = model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)

    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_name = [f"img_{batch_idx}_{i}.png" for i in range(batch_x.size(0))]

        raw = batch_x * std + mean
        raw = torch.clamp(raw, 0.0, 1.0)

        best_adv = raw.clone()
        best_loss = torch.full((raw.size(0),), -float("inf"), device=device)

        def _update_best(candidate, best_adv, best_loss):
            with torch.no_grad():
                per_sample = F.cross_entropy(
                    model((candidate - mean) / std), batch_y, reduction="none"
                )
                improved = per_sample > best_loss
                best_loss = torch.where(improved, per_sample, best_loss)
                best_adv[improved] = candidate[improved]
            return best_adv, best_loss

        restarts = max(1, int(getattr(args, "restarts", 1)))
        for _ in range(restarts):
            adv = raw + torch.empty_like(raw).uniform_(-args.eps, args.eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            for _ in range(args.iters):
                adv_norm = (adv - mean) / std
                adv_norm.requires_grad_(True)
                logits = model(adv_norm)
                per_sample = F.cross_entropy(logits, batch_y, reduction="none")
                loss = per_sample.mean()
                grad = torch.autograd.grad(loss, adv_norm, retain_graph=False, create_graph=False)[0]

                with torch.no_grad():
                    improved = per_sample.detach() > best_loss
                    best_loss = torch.where(improved, per_sample.detach(), best_loss)
                    best_adv[improved] = adv[improved].detach()

                adv = adv + args.step_size * grad.sign()
                adv = torch.max(torch.min(adv, raw + args.eps), raw - args.eps)
                adv = torch.clamp(adv, 0.0, 1.0)

            best_adv, best_loss = _update_best(adv, best_adv, best_loss)

        _save_images(best_adv, batch_name, images_dir)

    print(
        "PGD attack completed. "
        f"Images saved to {images_dir}, Metrics saved to {metrics_dir}"
    )

    _run_post_attack_eval(args, images_dir, metrics_dir)


def _run_post_attack_eval(args, images_dir, metrics_dir):
    ckpt_path = getattr(args, "ckpt_path", None)
    if not ckpt_path:
        print("[WARN] No ckpt_path provided; skipping evaluation.")
        return

    eval_script = os.path.join(os.path.dirname(__file__), "evaluate-attack.py")
    metrics_out = os.path.join(metrics_dir, "eval_metrics.json")
    eval_batch_size = getattr(args, "att_batch_size", None) or getattr(
        args, "batch_size", 64
    )

    cmd = [
        sys.executable,
        eval_script,
        "--images_dir",
        images_dir,
        "--ckpt_path",
        ckpt_path,
        "--dataset",
        args.dataset,
        "--ft_mode",
        args.ftmode,
        "--eval_batch_size",
        str(eval_batch_size),
        "--metrics_out",
        metrics_out,
        "--backbone",
        getattr(args, "backbone", "vitb16"),
    ]

    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except Exception as exc:
        print(f"[WARN] Post-attack evaluation failed: {exc}")
