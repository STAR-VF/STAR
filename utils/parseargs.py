import argparse
import yaml
import datetime
import os
import random
import numpy as np
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='STAR: robust adaptation of a pretrained ViT to a downstream task')

    parser.add_argument('--exp_mode', type=str, choices=['generate_target_model', 'attack_target_model'], required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./results')


    # for generate_target_model
    parser.add_argument("--source_model", type=str, choices=['vitb16', 'dinov2b14'], default='none')
    parser.add_argument("--target_model", type=str, choices=['vitb16', 'dinov2b14'], default='none')
    parser.add_argument("--source_dataset", type=str, choices=['ImageNet-1k', 'LVD-142M'], default='none')
    parser.add_argument("--target_dataset", type=str, choices=['cifar10', 'cifar100', 'oxford'], default='none')
    parser.add_argument('--ft_mode', type=str, choices=['fullft', 'lastft', 'adapter', 'lora', 'promptt', 'prefixt'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_batch_update', type=int, default=1,
        help='Number of mini-batches to accumulate gradients over before each optimizer step.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_schedule', type=str, choices=['cosine'], default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Linear LR warmup (in epochs) before cosine decay.')
    parser.add_argument('--grad_clip', type=float, default=0.0,
        help='Max global grad-norm before optimizer.step(). 0 disables clipping (default). '
             'Useful on small datasets where adversarial training can diverge in the first epoch.')
    parser.add_argument('--keep_last_ckpts', type=int, default=0,
        help='Keep only the N most recent per-epoch checkpoints (older ones are deleted after '
             'each save). 0 keeps every epoch (default). Each checkpoint bundles model '
             '+ optimizer + scheduler state; 2 is enough to resume.')
    parser.add_argument('--adv_eps_ramp_epochs', type=int, default=0,
        help='Linearly ramp the attack eps/step_size from 1/N to full over N epochs once the '
             'STAR turns on. 0 disables (default).')
    parser.add_argument('--adv_warmup_epochs', type=int, default=0,
        help='Train with clean CE only for the first N epochs, then enable STAR. '
             '0 disables (default). Avoids crafting PGD examples against a randomly-initialised head.')


    # for attack_target_model
    parser.add_argument('--attack_mode', type=str, choices=['none', 'PGD'], default='none')
    parser.add_argument('--ckpt_path', type=str, default=None)

    # PGD: training-time inner maximization and evaluation attack
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--step_size', type=float, default=2/255)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--min_value', type=float, default=0.0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--att_batch_size', type=int, default=64, help='Batch size of the post-attack evaluation.')


    # STAR loss
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature of the smoothness (KL) term.')
    parser.add_argument('--lambda_cle', type=float, default=1.0, help='Weight of the clean cross-entropy L_cle.')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Weight of the adversarial cross-entropy L_adv.')
    parser.add_argument('--lambda_smo', type=float, default=1.0, help='Weight of the decision-smoothness term L_smo.')
    parser.add_argument('--lambda_ali', type=float, default=1.0, help='Weight of the class-token alignment term L_ali.')

    args = parser.parse_args()
    return args


def check_arguments(args):
    """Validate CLI args against :func:`parse_arguments`."""

    exp_mode_choices = ('generate_target_model', 'attack_target_model')
    exp_mode = getattr(args, 'exp_mode', None)
    if exp_mode not in exp_mode_choices:
        raise ValueError(f"exp_mode must be one of: {exp_mode_choices}.")

    target_dataset = getattr(args, 'target_dataset', None)
    if target_dataset in (None, 'none'):
        raise ValueError("target_dataset must be 'cifar10', 'cifar100' or 'oxford'.")
    if target_dataset not in ('cifar10', 'cifar100', 'oxford'):
        raise ValueError("target_dataset must be either 'cifar10', 'cifar100' or 'oxford'.")
    args.target_dataset = target_dataset

    ft_mode_choices = ('fullft', 'lastft', 'adapter', 'lora', 'promptt', 'prefixt')
    if getattr(args, 'ft_mode', None) not in ft_mode_choices:
        raise ValueError(f"ft_mode must be one of: {ft_mode_choices}.")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    if args.n_batch_update <= 0:
        raise ValueError("n_batch_update must be > 0.")

    if getattr(args, 'grad_clip', 0.0) < 0:
        raise ValueError("grad_clip must be >= 0 (0 disables clipping).")

    if getattr(args, 'adv_warmup_epochs', 0) < 0:
        raise ValueError("adv_warmup_epochs must be >= 0.")

    if getattr(args, 'adv_eps_ramp_epochs', 0) < 0:
        raise ValueError("adv_eps_ramp_epochs must be >= 0.")

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0.")

    if getattr(args, 'lr', 0) <= 0:
        raise ValueError("lr must be > 0.")

    attack_mode_choices = ('none', 'PGD')
    if getattr(args, 'attack_mode', None) not in attack_mode_choices:
        raise ValueError(f"attack_mode must be one of: {attack_mode_choices}.")

    if args.eps < 0 or args.step_size < 0:
        raise ValueError("eps and step_size must be non-negative.")

    if args.alpha <= 0:
        raise ValueError("alpha must be > 0.")

    if args.iters <= 0:
        raise ValueError("iters must be > 0.")

    if args.min_value >= args.max_value:
        raise ValueError("min_value must be < max_value.")

    return args


def set_exp_result_dir(args):
    timestamp = os.environ.get('STAR_RUN_TIMESTAMP')
    if not timestamp:
        cur = datetime.datetime.now()
        timestamp = cur.strftime("%Y%m%d%H%M%S")

    save_path = args.save_path.rstrip("/")

    if args.exp_mode == 'generate_target_model':
        save_dir = f"{save_path}/{args.exp_mode}/source-{args.source_model}-{args.source_dataset}/target-{args.target_model}-{args.target_dataset}/ft-{args.ft_mode}/STAR"
        save_dir = f"{save_dir}/seed{args.seed:02d}/{timestamp}"
    elif args.exp_mode == 'attack_target_model':
        save_dir = f"{save_path}/{args.exp_mode}/source-{args.source_model}-{args.source_dataset}/target-{args.target_model}-{args.target_dataset}/ft-{args.ft_mode}/STAR/attack-{args.attack_mode}"
        save_dir = f"{save_dir}/seed{args.seed:02d}/{timestamp}"

    return save_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_argu():

    args = parse_arguments()
    check_arguments(args)
    print(args)

    set_seed(args.seed)
    args.save_path = set_exp_result_dir(args)
    os.makedirs(args.save_path, exist_ok=True)

    args_dictionary = vars(args)
    args_dictionary_yaml = yaml.dump(args_dictionary)
    with open(f'{args.save_path}/exp-command.yaml', "w") as exp_yaml:
        exp_yaml.write(args_dictionary_yaml)

    return args
