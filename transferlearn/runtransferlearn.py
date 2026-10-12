
import json
import os
import time

from transferlearn.finetune import ViTFinetuner


def _get_dataset_name(args):
    dataset = getattr(args, "target_dataset", None)
    if dataset in (None, "none"):
        dataset = getattr(args, "dataset", None)
    if dataset not in ("cifar10", "cifar100", "oxford"):
        raise ValueError(
            "Expected target dataset in {'cifar10', 'cifar100', 'oxford'} via "
            "args.target_dataset (or args.dataset)."
        )
    return dataset


def _get_backbone_name(args):
    backbone = getattr(args, "target_model", None)
    if backbone in (None, "none"):
        backbone = "vitb16"
    return backbone


def _train_by_mode(args, ft_mode):
    dataset = _get_dataset_name(args)
    device = getattr(args, "device", None)

    finetuner = ViTFinetuner(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        tune_mode=ft_mode,
        cfg=args,
        run_save_dir=getattr(args, "save_path", None),
        backbone=_get_backbone_name(args),
    )

    start_time = time.time()
    training_metrics = finetuner.train()
    total_time = time.time() - start_time
    training_metrics["total_time"] = total_time
    print(f"[INFO] Total training time (sec): {total_time:.2f}")

    run_id = os.environ.get("STAR_RUN_TIMESTAMP")
    if not run_id:
        run_id = os.path.basename(getattr(args, "save_path", "").rstrip("/"))
    if not run_id:
        run_id = time.strftime("%Y%m%d%H%M%S")
    epochs_tag = ""
    if getattr(args, "epochs", None):
        epochs_tag = f"{args.epochs}-epochs"
    metrics_dir = getattr(args, "save_path", None) or "logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_base = f"metrics_{ft_mode}_{dataset}_{run_id}"
    if epochs_tag:
        metrics_base = f"{metrics_base}_{epochs_tag}"
    metrics_path = os.path.join(metrics_dir, f"{metrics_base}.json")
    with open(metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(training_metrics, file_obj, indent=4)

    print(f"Saved training metrics to {metrics_path}")
    return finetuner


def train_fullft(args):
    return _train_by_mode(args, "fullft")


def train_lastft(args):
    return _train_by_mode(args, "lastft")


def train_adapter(args):
    return _train_by_mode(args, "adapter")


def train_lora(args):
    return _train_by_mode(args, "lora")


def train_promptt(args):
    return _train_by_mode(args, "promptt")


def train_prefixt(args):
    return _train_by_mode(args, "prefixt")


def evaluate_target_model(args, target_model):
    del args
    finetuner = target_model
    finetuner.test()


def run_generate_target_model(args):
    if args.ft_mode == "fullft":
        print("Full fine-tuning...")
        target_model = train_fullft(args)

    elif args.ft_mode == "lastft":
        print("Last layer fine-tuning...")
        target_model = train_lastft(args)

    elif args.ft_mode == "adapter":
        print("Adapter fine-tuning...")
        target_model = train_adapter(args)

    elif args.ft_mode == "lora":
        print("LoRA fine-tuning...")
        target_model = train_lora(args)

    elif args.ft_mode == "promptt":
        print("Prompt fine-tuning...")
        target_model = train_promptt(args)

    elif args.ft_mode == "prefixt":
        print("Prefix fine-tuning...")
        target_model = train_prefixt(args)

    else:
        raise ValueError(
            "Unsupported ft_mode. Expected one of: "
            "fullft, lastft, adapter, lora, promptt, prefixt"
        )

    evaluate_target_model(args, target_model)
