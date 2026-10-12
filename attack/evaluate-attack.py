import argparse
import json
import os
import re
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from transferlearn.finetune import ViTFinetuner


VIT_NORM_MEAN = (0.5, 0.5, 0.5)
VIT_NORM_STD = (0.5, 0.5, 0.5)
IMAGENET_NORM_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORM_STD = (0.229, 0.224, 0.225)


def _norm_stats(backbone):
	if backbone == "dinov2b14":
		return IMAGENET_NORM_MEAN, IMAGENET_NORM_STD
	return VIT_NORM_MEAN, VIT_NORM_STD


class AdvImageDataset(Dataset):
	def __init__(self, items, transform):
		self.items = items
		self.transform = transform

	def __len__(self):
		return len(self.items)

	def __getitem__(self, idx):
		image_path, label = self.items[idx]
		image = Image.open(image_path).convert("RGB")
		if self.transform is not None:
			image = self.transform(image)
		return image, label


def _load_state_dict_model(args, device):
	finetuner = ViTFinetuner(
		dataset=args.dataset,
		batch_size=args.eval_batch_size,
		epochs=1,
		lr=3e-4,
		device=device,
		tune_mode=args.ft_mode,
		backbone=args.backbone,
	)

	checkpoint = torch.load(args.ckpt_path, map_location=device)
	if isinstance(checkpoint, dict):
		if "state_dict" in checkpoint:
			state_dict = checkpoint["state_dict"]
		elif "model" in checkpoint:
			state_dict = checkpoint["model"]
		else:
			state_dict = checkpoint
	else:
		state_dict = checkpoint

	missing, unexpected = finetuner.model.load_state_dict(state_dict, strict=False)
	if args.backbone != "vitb16" and (missing or unexpected):
		raise RuntimeError(
			f"Checkpoint does not match backbone '{args.backbone}': "
			f"missing={missing[:5]} unexpected={unexpected[:5]}")
	finetuner.model.to(device)
	finetuner.model.eval()
	return finetuner.model


def _build_dataset(dataset_name, backbone="vitb16"):
	norm_mean, norm_std = _norm_stats(backbone)
	transform = transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(norm_mean, norm_std),
		]
	)
	data_root = os.path.join(os.path.dirname(__file__), "..", "data")
	data_root = os.path.abspath(data_root)

	if dataset_name == "cifar10":
		return datasets.CIFAR10(
			root=data_root,
			train=False,
			download=True,
			transform=transform,
		)
	if dataset_name == "cifar100":
		return datasets.CIFAR100(
			root=data_root,
			train=False,
			download=True,
			transform=transform,
		)
	if dataset_name == "oxford":
		return datasets.OxfordIIITPet(
			root=data_root,
			split="test",
			target_types="category",
			download=True,
			transform=transform,
		)
	raise ValueError(f"Unsupported dataset: {dataset_name}")


def _parse_image_indices(images_dir):
	pattern = re.compile(r"img_(\d+)_(\d+)\.png$")
	items = []
	max_i = -1

	for name in os.listdir(images_dir):
		match = pattern.match(name)
		if not match:
			continue
		batch_idx = int(match.group(1))
		image_idx = int(match.group(2))
		max_i = max(max_i, image_idx)
		items.append((batch_idx, image_idx, os.path.join(images_dir, name)))

	if not items:
		raise ValueError(f"No images found in {images_dir}")

	items.sort(key=lambda x: (x[0], x[1]))
	inferred_batch_size = max_i + 1
	return items, inferred_batch_size


def _build_adv_items(dataset, images_dir):
	items, batch_size = _parse_image_indices(images_dir)

	if hasattr(dataset, "targets"):
		labels = dataset.targets
	else:
		labels = [dataset[i][1] for i in range(len(dataset))]

	adv_items = []
	adv_indices = []
	for batch_idx, image_idx, image_path in items:
		global_idx = batch_idx * batch_size + image_idx
		if global_idx >= len(labels):
			continue
		adv_indices.append(global_idx)
		adv_items.append((image_path, int(labels[global_idx])))

	if not adv_items:
		raise ValueError("No valid image indices matched the dataset.")

	return adv_items, adv_indices


def _evaluate(model, loader, device):
	correct_top1 = 0
	correct_top5 = 0
	total = 0

	with torch.no_grad():
		for inputs, labels in loader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			logits = model(inputs)
			_, top5 = logits.topk(5, dim=1, largest=True, sorted=True)

			correct_top1 += (top5[:, 0] == labels).sum().item()
			correct_top5 += (top5 == labels.view(-1, 1)).any(dim=1).sum().item()
			total += labels.size(0)

	if total == 0:
		return 0.0, 0.0, 0

	top1 = 100.0 * correct_top1 / total
	top5 = 100.0 * correct_top5 / total
	return top1, top5, total


def main():
	parser = argparse.ArgumentParser(
		description="Evaluate clean and adversarial accuracy for an attack run."
	)
	parser.add_argument("--images_dir", required=True)
	parser.add_argument("--ckpt_path", required=True)
	parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "oxford"])
	parser.add_argument("--ft_mode", required=True)
	parser.add_argument("--eval_batch_size", type=int, default=64)
	parser.add_argument("--metrics_out", default="")
	parser.add_argument("--backbone", default="vitb16", choices=["vitb16", "dinov2b14"])

	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = _load_state_dict_model(args, device)

	dataset = _build_dataset(args.dataset, args.backbone)
	adv_items, adv_indices = _build_adv_items(dataset, args.images_dir)

	adv_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(*_norm_stats(args.backbone)),
		]
	)
	adv_dataset = AdvImageDataset(adv_items, adv_transform)
	adv_loader = DataLoader(
		adv_dataset,
		batch_size=args.eval_batch_size,
		shuffle=False,
		num_workers=0,
	)

	clean_loader = DataLoader(
		dataset,
		batch_size=args.eval_batch_size,
		shuffle=False,
		num_workers=0,
	)

	clean_top1, clean_top5, clean_total = _evaluate(model, clean_loader, device)
	adv_top1, adv_top5, adv_total = _evaluate(model, adv_loader, device)

	metrics = {
		"dataset": args.dataset,
		"ft_mode": args.ft_mode,
		"clean_total_samples": clean_total,
		"adv_total_samples": adv_total,
		"clean_top1": clean_top1,
		"clean_top5": clean_top5,
		"adv_top1": adv_top1,
		"adv_top5": adv_top5,
		"images_dir": os.path.abspath(args.images_dir),
		"ckpt_path": os.path.abspath(args.ckpt_path),
	}

	metrics_out = args.metrics_out
	if not metrics_out:
		attack_dir = os.path.abspath(os.path.dirname(args.images_dir))
		metrics_dir = os.path.join(attack_dir, "metrics")
		os.makedirs(metrics_dir, exist_ok=True)
		metrics_out = os.path.join(metrics_dir, "eval_metrics.json")

	with open(metrics_out, "w", encoding="utf-8") as file_obj:
		json.dump(metrics, file_obj, indent=4)

	print("Evaluation complete.")
	print(f"Clean Top-1: {clean_top1:.2f}% | Clean Top-5: {clean_top5:.2f}%")
	print(f"Adv   Top-1: {adv_top1:.2f}% | Adv   Top-5: {adv_top5:.2f}%")
	print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
	main()
