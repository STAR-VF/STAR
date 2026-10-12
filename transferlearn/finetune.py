import os
import math
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import glob
import re
import torch.optim.lr_scheduler
import torch.nn.functional as F

VIT_NORM_MEAN = (0.5, 0.5, 0.5)
VIT_NORM_STD = (0.5, 0.5, 0.5)
IMAGENET_NORM_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORM_STD = (0.229, 0.224, 0.225)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RESULTS_CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "results", "checkpoints")

DINOV2_VITB14_WEIGHTS = os.path.join(DATA_ROOT, "checkpoints", "dinov2_vitb14_pretrain.pth")

BACKBONE_VITB16 = "vitb16"
BACKBONE_DINOV2B14 = "dinov2b14"
SUPPORTED_BACKBONES = (BACKBONE_VITB16, BACKBONE_DINOV2B14)


def backbone_norm_stats(backbone):
    """Input mean/std the given pretrained backbone expects."""
    backbone = (backbone or BACKBONE_VITB16).lower()
    if backbone == BACKBONE_DINOV2B14:
        return IMAGENET_NORM_MEAN, IMAGENET_NORM_STD
    return VIT_NORM_MEAN, VIT_NORM_STD


class _LayerScaledBlockForward:

    def __init__(self, block):
        self.block = block

    def __call__(self, x):
        b = self.block
        x = x + b.drop_path(b.ls1_gamma * b.attn(b.norm1(x)))
        x = x + b.drop_path(b.ls2_gamma * b.mlp(b.norm2(x)))
        return x


def _build_dinov2_vitb14(img_size=224, weights_path=DINOV2_VITB14_WEIGHTS):
    from functools import partial
    from timm.models.vision_transformer import VisionTransformer, resize_pos_embed

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DINOv2 weights not found at {weights_path}. Download with:\n"
            "  curl -L -o data/checkpoints/dinov2_vitb14_pretrain.pth "
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
        )

    model = VisionTransformer(
        img_size=img_size, patch_size=14, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True, num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    sd = torch.load(weights_path, map_location="cpu")
    sd.pop("mask_token", None)

    sd["pos_embed"] = resize_pos_embed(sd["pos_embed"], model.pos_embed, num_tokens=1)

    for i, block in enumerate(model.blocks):
        block.ls1_gamma = nn.Parameter(sd.pop(f"blocks.{i}.ls1.gamma"))
        block.ls2_gamma = nn.Parameter(sd.pop(f"blocks.{i}.ls2.gamma"))
        block.forward = _LayerScaledBlockForward(block)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith(("ls1_gamma", "ls2_gamma"))]
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv2 weight loading mismatch. missing={missing} unexpected={unexpected}"
        )
    print(f"[INFO] Loaded DINOv2 ViT-B/14 weights from {weights_path} "
          f"(pos_embed resampled to {model.patch_embed.grid_size})")
    return model


def build_backbone(backbone=BACKBONE_VITB16):
    backbone = (backbone or BACKBONE_VITB16).lower()
    if backbone == BACKBONE_VITB16:
        return timm.create_model('vit_base_patch16_224', pretrained=True)
    if backbone == BACKBONE_DINOV2B14:
        model = _build_dinov2_vitb14()
        model.head = nn.Linear(model.embed_dim, 1000)
        return model
    raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of {SUPPORTED_BACKBONES}")


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_down = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, original_linear.out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x):
        return self.original_linear(x) + self.lora_up(self.lora_down(x)) * self.scaling


class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, dim)

    def forward(self, x):
        return x + self.up_proj(self.relu(self.down_proj(x)))


class PromptEmbedding(nn.Module):
    def __init__(self, prompt_len, embed_dim):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_len, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        prompt = self.prompt_embeddings.expand(B, -1, -1)
        return torch.cat([prompt, x], dim=1)


class PrefixAttention(nn.Module):
    def __init__(self, attn_module, prefix_len=10, dim=768, num_heads=12):
        super().__init__()
        self.attn = attn_module
        self.prefix_len = prefix_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.prefix_keys = nn.Parameter(torch.randn(prefix_len, dim))
        self.prefix_values = nn.Parameter(torch.randn(prefix_len, dim))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.attn.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        prefix_k = self.prefix_keys.unsqueeze(0).expand(B, -1, -1)
        prefix_v = self.prefix_values.unsqueeze(0).expand(B, -1, -1)
        prefix_k = prefix_k.reshape(B, self.prefix_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        prefix_v = prefix_v.reshape(B, self.prefix_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = torch.cat([prefix_k, k], dim=2)
        v = torch.cat([prefix_v, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.attn.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class ViTFinetuner:
    def __init__(
        self,
        dataset='cifar10',
        batch_size=256,
        epochs=200,
        lr=3e-4,
        device=None,
        tune_mode='last',
        cfg=None,
        run_save_dir=None,
        backbone=BACKBONE_VITB16,
    ):
        self.dataset_name = dataset
        self.backbone = (backbone or BACKBONE_VITB16).lower()
        self.norm_mean_tuple, self.norm_std_tuple = backbone_norm_stats(self.backbone)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tune_mode = tune_mode
        self.cfg = cfg
        self.run_save_dir = run_save_dir
        self.n_batch_update = max(1, int(getattr(cfg, 'n_batch_update', 1)))
        print(self.tune_mode)

        self.num_classes = {'cifar10': 10, 'cifar100': 100, 'oxford': 37}.get(dataset, 100)
        self._prepare_data()
        self._build_model()
        self.norm_mean = torch.tensor(self.norm_mean_tuple, device=self.device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(self.norm_std_tuple, device=self.device).view(1, 3, 1, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.05)

        self.warmup_epochs = max(0, int(getattr(cfg, 'warmup_epochs', 0) or 0))
        if self.warmup_epochs > 0 and self.warmup_epochs < self.epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - self.warmup_epochs,
            )
            self.schedule = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

        self.adv_eps = float(getattr(cfg, 'eps', 8.0 / 255.0))
        self.adv_alpha = float(
            getattr(cfg, 'step_size', getattr(cfg, 'alpha', 2.0 / 255.0))
        )
        self.adv_steps = int(getattr(cfg, 'iters', getattr(cfg, 'steps', 10)))
        self.temp = float(getattr(cfg, 'temperature', 1.0))
        self.grad_clip = float(getattr(cfg, 'grad_clip', 0.0) or 0.0)
        self.adv_warmup_epochs = max(0, int(getattr(cfg, 'adv_warmup_epochs', 0) or 0))
        self.adv_eps_ramp_epochs = max(0, int(getattr(cfg, 'adv_eps_ramp_epochs', 0) or 0))
        self._clean_only_phase = False
        self._eps_scale = 1.0
        self.keep_last_ckpts = max(0, int(getattr(cfg, 'keep_last_ckpts', 0) or 0))
        self.lambda_cle = float(getattr(cfg, 'lambda_cle', 1.0))
        self.lambda_adv = float(getattr(cfg, 'lambda_adv', 1.0))
        self.lambda_smo = float(getattr(cfg, 'lambda_smo', 1.0))
        self.lambda_ali = float(getattr(cfg, 'lambda_ali', 1.0))

    def _get_transforms(self):
        normalize = transforms.Normalize(self.norm_mean_tuple, self.norm_std_tuple)
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, test_transform

    def _prepare_data(self):
        train_transform, test_transform = self._get_transforms()
        if self.dataset_name == 'oxford':
            full_train = datasets.OxfordIIITPet(
                root=DATA_ROOT, split='trainval', target_types='category',
                download=True, transform=train_transform)
            self.test_dataset = datasets.OxfordIIITPet(
                root=DATA_ROOT, split='test', target_types='category',
                download=True, transform=test_transform)
        else:
            DatasetClass = datasets.CIFAR10 if self.dataset_name == 'cifar10' else datasets.CIFAR100
            full_train = DatasetClass(
                root=DATA_ROOT, train=True, download=True, transform=train_transform)
            self.test_dataset = DatasetClass(
                root=DATA_ROOT, train=False, download=True, transform=test_transform)
        val_size = int(0.1 * len(full_train))
        train_size = len(full_train) - val_size
        self.train_dataset, self.val_dataset = random_split(full_train, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def _build_model(self):
        self.model = build_backbone(self.backbone)
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        if self.tune_mode == 'lastft':
            for param in self.model.head.parameters():
                param.requires_grad = True

        elif self.tune_mode == 'fullft':
            for param in self.model.parameters():
                param.requires_grad = True
                
        elif self.tune_mode == 'adapter':
            self._add_adapters(bottleneck=64)
            self._freeze_backbone_train_adapters()

        elif self.tune_mode == 'promptt':
            self._add_prompt_tuning(num_prompts=20)
            for param in self.model.head.parameters():
                param.requires_grad = True

        elif self.tune_mode == 'prefixt':
            self._add_prefix_tuning(prefix_len=10)
            for param in self.model.head.parameters():
                param.requires_grad = True
        
        elif self.tune_mode == 'lora':
            self._add_lora(r=4, alpha=16)
            for param in self.model.parameters():
                param.requires_grad = False
            for block in self.model.blocks:
                for param in block.attn.qkv.lora_down.parameters():
                    param.requires_grad = True
                for param in block.attn.qkv.lora_up.parameters():
                    param.requires_grad = True
            for param in self.model.head.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Invalid tune_mode: {self.tune_mode}")

        self.model = self.model.to(self.device)

    def _add_adapters(self, bottleneck=64):
        for block in self.model.blocks:
            dim = block.mlp.fc2.out_features
            block.adapter = Adapter(dim, bottleneck).to(self.device)

            original_forward = block.forward

            def new_forward(x, original_forward=original_forward, adapter=block.adapter):
                x = original_forward(x)
                x = adapter(x)
                return x

            block.forward = new_forward

    def _freeze_backbone_train_adapters(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for block in self.model.blocks:
            for param in block.adapter.parameters():
                param.requires_grad = True

        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def _add_prompt_tuning(self, num_prompts=20):
        dim = self.model.pos_embed.shape[-1]
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts, dim))

        def interpolate_pos_embed(pos_embed, x_len, num_prompts):
            cls_token = pos_embed[:, 0:1, :]
            patch_pos_embed = pos_embed[:, 1:, :]
            num_patches = patch_pos_embed.shape[1]
            dim = patch_pos_embed.shape[2]
            H = W = int(num_patches ** 0.5)

            patch_pos_embed = patch_pos_embed.reshape(1, H, W, dim).permute(0, 3, 1, 2)

            new_H = new_W = int(x_len ** 0.5)
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_H, new_W), mode='bicubic', align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_H * new_W, dim)

            prompt_pos_embed = torch.zeros(1, num_prompts, dim, device=pos_embed.device, dtype=pos_embed.dtype)

            return torch.cat((cls_token, prompt_pos_embed, patch_pos_embed), dim=1)

        def new_forward_features(x):
            B = x.shape[0] 
            x = self.model.patch_embed(x)
            num_patches = x.shape[1]

            cls_token = self.model.cls_token.expand(B, -1, -1)
            prompts = self.prompt_embeddings.expand(B, -1, -1).to(x.device)

            x = torch.cat((cls_token, prompts, x), dim=1)
            
            pos_embed = interpolate_pos_embed(self.model.pos_embed, num_patches, self.prompt_embeddings.shape[1])
            pos_embed = pos_embed.to(x.device)

            x = x + pos_embed
            x = self.model.pos_drop(x)

            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
            
            return x[:, 0, :]

        self.model.forward_features = new_forward_features
        self.model.forward = lambda x: self.model.head(self.model.forward_features(x))

        self.model.prompt_embeddings = self.prompt_embeddings
        for param in self.model.head.parameters():
            param.requires_grad = True
        self.prompt_embeddings.requires_grad = True


    
    def _add_prefix_tuning(self, prefix_len=10):
        dim = self.model.blocks[0].attn.qkv.in_features
        for i, block in enumerate(self.model.blocks):
            block.attn = PrefixAttention(block.attn, prefix_len=prefix_len, dim=dim)

    def _add_lora(self, r=4, alpha=16):
        for block in self.model.blocks:
            
            attn = block.attn
            
            attn.qkv = LoRALinear(attn.qkv, r=r, alpha=alpha).to(self.device)

    def _normalize(self, x_raw):
        return (x_raw - self.norm_mean) / self.norm_std

    def _denormalize(self, x_norm):
        return x_norm * self.norm_std + self.norm_mean

    def _extract_features(self, x_norm):
        if hasattr(self.model, 'forward_features'):
            feats = self.model.forward_features(x_norm)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if isinstance(feats, torch.Tensor):
                if feats.ndim == 3:
                    return feats[:, 0, :]
                return feats
        return self.model(x_norm)

    def _pgd_linf_attack(self, x_norm, y):
        """PGD attack in raw image space to preserve correct eps/alpha semantics."""
        was_training = self.model.training
        self.model.eval()

        x_raw = self._denormalize(x_norm).detach().clamp(0.0, 1.0)
        x_adv = (x_raw + 0.001 * torch.randn_like(x_raw)).clamp(0.0, 1.0)
        eps = self.adv_eps * self._eps_scale
        alpha = self.adv_alpha * self._eps_scale

        for _ in range(self.adv_steps):
            x_adv.requires_grad_(True)
            logits_adv = self.model(self._normalize(x_adv))
            loss_adv = self.criterion(logits_adv, y)
            grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_raw, min=-eps, max=eps)
            x_adv = (x_raw + delta).clamp(0.0, 1.0)

        self.model.train(was_training)
        return self._normalize(x_adv.detach())

    def _compute_loss(self, images, labels):
        """Return the STAR loss and its terms for logging."""
        if self._clean_only_phase:
            return self.criterion(self.model(images), labels), None

        adv_images = self._pgd_linf_attack(images, labels)

        # class tokens z = h(x), z* = h(x*)
        feat_clean = self._extract_features(images)
        feat_adv = self._extract_features(adv_images)
        logits_clean = self.model.head(feat_clean)
        logits_adv = self.model.head(feat_adv)

        # decision level: clean and adversarial cross-entropy
        l_cle = self.criterion(logits_clean, labels)
        l_adv = self.criterion(logits_adv, labels)

        # decision smoothness: KL between clean and midpoint-token predictions
        t = max(self.temp, 1e-6)
        alpha_mix = 0.5
        feat_mix = alpha_mix * feat_clean + (1.0 - alpha_mix) * feat_adv
        logits_mix = self.model.head(feat_mix)
        clean_log_prob = F.log_softmax(logits_clean / t, dim=1)
        mix_prob = F.softmax(logits_mix / t, dim=1)
        l_smo = F.kl_div(clean_log_prob, mix_prob, reduction='batchmean') * (t * t)

        # representation level: class-token alignment
        l_ali = (1.0 - F.cosine_similarity(feat_clean, feat_adv, dim=1)).mean()

        total = (
            self.lambda_cle * l_cle
            + self.lambda_adv * l_adv
            + self.lambda_smo * l_smo
            + self.lambda_ali * l_ali
        )
        terms = {
            'L_cle': float(l_cle.detach().item()),
            'L_adv': float(l_adv.detach().item()),
            'L_smo': float(l_smo.detach().item()),
            'L_ali': float(l_ali.detach().item()),
        }
        return total, terms

    def train(self):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        epoch_times = []
        train_l_cle = []
        train_l_adv = []
        train_l_smo = []
        train_l_ali = []

        run_base = (
            f"vit_finetuned_{self.dataset_name}_{self.tune_mode}_star"
        )
        if self.run_save_dir:
            save_root = self.run_save_dir
        else:
            save_root = os.path.join(
                RESULTS_CHECKPOINT_ROOT,
                f"{self.dataset_name}-finetuned",
            )
        os.makedirs(save_root, exist_ok=True)

        start_epoch = 0
        pattern = os.path.join(save_root, f"{run_base}_epoch*.pth")
        ckpts = glob.glob(pattern)
        if ckpts:
            max_epoch = 0
            max_path = None
            for p in ckpts:
                m = re.search(r"_epoch(\d+)\.pth$", p)
                if m:
                    try:
                        e = int(m.group(1))
                    except ValueError:
                        continue
                    if e > max_epoch:
                        max_epoch = e
                        max_path = p
            if max_path and max_epoch > 0:
                try:
                    ckpt = torch.load(max_path, map_location=self.device)
                    if 'model_state' in ckpt:
                        self.model.load_state_dict(ckpt['model_state'])
                    if 'optim_state' in ckpt:
                        try:
                            self.optimizer.load_state_dict(ckpt['optim_state'])
                        except Exception:
                            print(f"[WARN] Couldn't fully restore optimizer state from {max_path}")
                    if 'sched_state' in ckpt:
                        try:
                            self.schedule.load_state_dict(ckpt['sched_state'])
                        except Exception:
                            print(f"[WARN] Couldn't fully restore scheduler state from {max_path}")
                    start_epoch = max_epoch
                    print(f"[INFO] Resuming from checkpoint {max_path} at epoch {start_epoch + 1}")
                except Exception as exc:
                    print(f"[WARN] Failed to load checkpoint {max_path}: {exc}")

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            was_clean_only = self._clean_only_phase
            self._clean_only_phase = epoch < self.adv_warmup_epochs
            if was_clean_only and not self._clean_only_phase:
                print(f"[INFO] Epoch {epoch + 1}: clean warmup over, enabling STAR.")
            if self.adv_eps_ramp_epochs > 0 and not self._clean_only_phase:
                adv_epoch = epoch - self.adv_warmup_epochs
                prev_scale = self._eps_scale
                self._eps_scale = min(1.0, (adv_epoch + 1) / self.adv_eps_ramp_epochs)
                if self._eps_scale != prev_scale:
                    print(f"[INFO] Epoch {epoch + 1}: eps ramp {self._eps_scale:.3f} "
                          f"(eps={self.adv_eps * self._eps_scale:.5f})")
            start_time = time.time()
            total_loss, correct, total = 0, 0, 0

            self.optimizer.zero_grad()

            for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
                images, labels = images.to(self.device), labels.to(self.device)

                loss, extra_terms = self._compute_loss(images, labels)
                (loss / self.n_batch_update).backward()

                if (
                    batch_idx % self.n_batch_update == 0
                    or batch_idx == len(self.train_loader)
                ):
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.grad_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                outputs = self.model(images)

                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                if extra_terms is not None:
                    train_l_cle.append(extra_terms['L_cle'])
                    train_l_adv.append(extra_terms['L_adv'])
                    train_l_smo.append(extra_terms['L_smo'])
                    train_l_ali.append(extra_terms['L_ali'])

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            train_acc = 100 * correct / total
            train_loss = total_loss / total
            train_accs.append(train_acc)
            train_losses.append(train_loss)

            val_loss, val_acc = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            self.schedule.step()
            current_lr = self.schedule.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                f"lr={current_lr:.6g} | time={epoch_time:.2f}s"
            )
            try:
                ckpt_fname = os.path.join(save_root, f"{run_base}_epoch{epoch + 1}.pth")
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state': self.model.state_dict(),
                        'optim_state': self.optimizer.state_dict(),
                        'sched_state': self.schedule.state_dict(),
                    },
                    ckpt_fname,
                )
                print(f"[INFO] Saved checkpoint: {ckpt_fname}")
                if self.keep_last_ckpts > 0:
                    kept = []
                    for old_p in glob.glob(os.path.join(save_root, f"{run_base}_epoch*.pth")):
                        m = re.search(r"_epoch(\d+)\.pth$", old_p)
                        if m:
                            kept.append((int(m.group(1)), old_p))
                    for _, old_p in sorted(kept, reverse=True)[self.keep_last_ckpts:]:
                        try:
                            os.remove(old_p)
                        except OSError as exc:
                            print(f"[WARN] Couldn't prune {old_p}: {exc}")
            except Exception as exc:
                print(f"[WARN] Failed to write checkpoint for epoch {epoch + 1}: {exc}")
        metrics = {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
            "epoch_time": epoch_times
        }
        if train_l_cle:
            metrics["star_terms"] = {
                "L_cle_mean": float(sum(train_l_cle) / len(train_l_cle)),
                "L_adv_mean": float(sum(train_l_adv) / len(train_l_adv)),
                "L_smo_mean": float(sum(train_l_smo) / len(train_l_smo)),
                "L_ali_mean": float(sum(train_l_ali) / len(train_l_ali)),
            }
        try:
            run_id = os.environ.get("STAR_RUN_TIMESTAMP")
            tag_parts = []
            if self.epochs:
                tag_parts.append(f"{self.epochs}-epochs")
            if run_id:
                tag_parts.append(run_id)
            checkpoint_tag = "_".join(tag_parts) if tag_parts else None
            if checkpoint_tag:
                final_name = f"{run_base}_{checkpoint_tag}.pth"
                final_path = os.path.join(save_root, final_name)
                latest_ckpts = glob.glob(os.path.join(save_root, f"{run_base}_epoch*.pth"))
                if latest_ckpts:
                    latest = max(latest_ckpts, key=os.path.getmtime)
                    try:
                        import shutil

                        shutil.copy2(latest, final_path)
                        print(f"[INFO] Copied final checkpoint to {final_path}")
                    except Exception as exc:
                        print(f"[WARN] Failed to copy final checkpoint: {exc}")
        except Exception:
            pass

        return metrics
        
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        val_loss = total_loss / total
        return val_loss, val_acc
    
    def test(self):
        self.model.eval()
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                _, pred_top5 = outputs.topk(5, dim=1)
                total += labels.size(0)
                top1_correct += (pred_top5[:, 0] == labels).sum().item()
                top5_correct += sum([labels[i] in pred_top5[i] for i in range(labels.size(0))])

        top1_acc = 100 * top1_correct / total
        top5_acc = 100 * top5_correct / total
        print(f"Test Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"Test Top-5 Accuracy: {top5_acc:.2f}%")

        tag_parts = []
        if self.epochs:
            tag_parts.append(f"{self.epochs}-epochs")
        run_id = os.environ.get("STAR_RUN_TIMESTAMP")
        if not run_id and self.run_save_dir:
            run_id = os.path.basename(self.run_save_dir.rstrip("/"))
        if run_id:
            tag_parts.append(run_id)
        checkpoint_tag = "_".join(tag_parts) if tag_parts else None
        def _save_model_object_safely(model_obj, path):
            try:
                torch.save(model_obj, path)
                print(f"Saved model object to {path}")
            except Exception as exc:
                print(
                    "[WARN] Skipping model-object save "
                    f"({path}): {exc}"
                )

        if self.run_save_dir:
            save_root = self.run_save_dir
        else:
            save_root = os.path.join(
                RESULTS_CHECKPOINT_ROOT,
                f"{self.dataset_name}-finetuned",
            )
        os.makedirs(save_root, exist_ok=True)

        run_base = (
            f"vit_finetuned_{self.dataset_name}_{self.tune_mode}_star"
        )
        if checkpoint_tag:
            run_base = f"{run_base}_{checkpoint_tag}"
        run_pth = os.path.join(save_root, f"{run_base}.pth")
        run_pkl = os.path.join(save_root, f"{run_base}.pkl")
        torch.save(self.model.state_dict(), run_pth)
        _save_model_object_safely(self.model, run_pkl)
        print(f"Saved run state_dict to {run_pth}")
