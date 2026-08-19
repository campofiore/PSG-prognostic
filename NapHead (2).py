from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from typing import Dict, List, Optional
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex

from NapFM3 import *               # provides SleepFoundationModel (+ PSGEncoder etc.)
from AdamMuon import AdamMuon


# =========================================================
# GROUPS
# =========================================================
GROUPS = {
    "ischemic": [
        "Myocardial Infarction", "Ischemic Heart Disease", "Angina",
        "General Atherosclerosis",
    ],
    "arrhythmia": ["Atrial Fibrilation and Flutter"],
    "heart_failure": ["Heart Failure", "Pulmonary Heart Disease"],
    "hemodynamic": ["Hypertension", "Hypotension"],
    "metabolic": ["Type 2 Diabetes", "Chronic Kidney Disease", "Dementia"],
    "mortality": ["death"],
}


# =========================================================
# LoRA
# =========================================================
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features

        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # A small, B zero -> LoRA starts as a no-op
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale = alpha / rank

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora

    def extra_repr(self):
        return (f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
                f"rank={self.lora_A.shape[0]}")


def apply_lora_to_backbone(
    backbone: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_keywords: tuple = (
        "enc.proj",                 # SpectrogramEncoder output projection
        "spatial_pool",             # AttentionPooling: self_attn.out_proj + FFN linears
        "cross_modal_fusion.layer", # NEW fusion: TransformerEncoderLayer linears
                                    #   (was "cross_modal_fusion.cross_attn"/".ff" — stale)
    ),
    skip_keywords: tuple = (
        "norm",
        "private_projector",        # unused downstream (we consume *_raw features)
        "shared_projector",
    ),
) -> nn.Module:
    named = dict(backbone.named_modules())
    replaced, skipped = 0, 0
    for full_name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(sk in full_name for sk in skip_keywords):
            skipped += 1
            continue
        if not any(kw in full_name for kw in target_keywords):
            skipped += 1
            continue
        parts = full_name.split(".")
        parent = backbone if len(parts) == 1 else named[".".join(parts[:-1])]
        setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))
        replaced += 1
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"LoRA: replaced {replaced} linears | skipped {skipped}")
    print(f"Backbone trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return backbone


# =========================================================
# MAMBA (your block, unchanged) + lean bidirectional wrapper
# =========================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4,
                 expand: int = 2, dt_min: float = 0.001, dt_max: float = 0.1,
                 scan_chunk: int = 512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.scan_chunk = scan_chunk

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )

        self.dt_rank = max(1, math.ceil(math.log2(d_model)))
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_proj.bias = nn.Parameter(torch.log(torch.expm1(dt)))
        self.dt_proj.bias._no_weight_decay = True

        A = (torch.arange(1, d_state + 1, dtype=torch.float32)
             .unsqueeze(0).expand(self.d_inner, -1))
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, x, delta, B, C):
        B_batch, L, D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())                       # [D, N]

        y_chunks = []
        log_cum_A_prev = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)

        for start in range(0, L, self.scan_chunk):
            end = min(start + self.scan_chunk, L)
            x_c = x[:, start:end]
            delta_c = delta[:, start:end]
            B_c = B[:, start:end]
            C_c = C[:, start:end]

            deltaA = torch.einsum('bld,dn->bldn', delta_c, A)
            deltaB_x = (torch.einsum('bld,bln->bldn', delta_c, B_c)
                        * x_c.unsqueeze(-1))

            log_cum_A = torch.cumsum(deltaA, dim=1) + log_cum_A_prev.unsqueeze(1)
            log_cum_A = torch.clamp(log_cum_A, min=-30.0, max=30.0)
            cum_A = torch.exp(log_cum_A)

            scaled = deltaB_x / cum_A.clamp(min=1e-12)
            h = torch.cumsum(scaled, dim=1) * cum_A

            y_c = torch.einsum('bldn,bln->bld', h, C_c)
            y_c = y_c + self.D.unsqueeze(0).unsqueeze(0) * x_c
            y_chunks.append(y_c)

            log_cum_A_prev = log_cum_A[:, -1]

        return torch.cat(y_chunks, dim=1)

    def forward(self, u):
        residual = u
        u = self.norm(u)

        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :u.shape[1]]
        x_conv = F.silu(x_conv.transpose(1, 2))

        x_db = self.x_proj(x_conv)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))

        with torch.amp.autocast('cuda', enabled=False):
            y = self._selective_scan(x_conv.float(), delta.float(),
                                     B.float(), C.float())
        y = y.to(x.dtype)

        y = y * F.silu(z)
        y = self.out_proj(y)
        return y + residual


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BidirectionalMambaBlock(nn.Module):
    """Lean bidirectional block: shared FFN across directions + concat-merge.
    Bidirectional is justified here because we do offline full-night inference and
    then GLOBAL pool -> every position should carry full-night context."""
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1,
                 share_ffn: bool = True, merge: str = "concat"):
        super().__init__()
        self.merge = merge
        self.mamba_fwd = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_rev = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln1_rev = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.ffn_rev = self.ffn if share_ffn else FeedForward(d_model, dropout=dropout)
        if merge == "concat":
            self.merge_proj = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _branch(self, x, mamba, ln1, flip):
        x_in = torch.flip(x, dims=[1]) if flip else x
        y = self.dropout(mamba(x_in))
        if flip:
            y = torch.flip(y, dims=[1])
        return ln1(x + y)

    def forward(self, x):
        fwd = self._branch(x, self.mamba_fwd, self.ln1, flip=False)
        rev = self._branch(x, self.mamba_rev, self.ln1_rev, flip=True)
        merged = (self.merge_proj(torch.cat([fwd, rev], dim=-1))
                  if self.merge == "concat" else 0.5 * (fwd + rev))
        return self.ln2(merged + self.dropout(self.ffn(merged)))


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model=d_model, **block_kwargs)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)


# =========================================================
# Demographic (age) risk tower — additive log-hazard baseline
# =========================================================
class AgeRiskTower(nn.Module):
    def __init__(self, n_tasks: int, n_groups: int, hidden: int = 64,
                 n_rbf: int = 8, age_min: float = 18.0, age_max: float = 95.0):
        super().__init__()
        self.register_buffer("rbf_centers", torch.linspace(age_min, age_max, n_rbf))
        self.rbf_sigma = (age_max - age_min) / n_rbf
        in_dim = n_rbf + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.task_out = nn.Linear(hidden, n_tasks)
        self.group_out = nn.Linear(hidden, n_groups) if n_groups > 0 else None

    def _features(self, age):
        a = age.float().view(-1, 1)
        z = (a - 50.0) / 20.0
        rbf = torch.exp(-((a - self.rbf_centers.view(1, -1)) ** 2)
                        / (2 * self.rbf_sigma ** 2))
        return torch.cat([z, z ** 2, rbf], dim=1)

    def forward(self, age):
        h = self.net(self._features(age))
        groups = self.group_out(h) if self.group_out is not None else None
        return self.task_out(h), groups


# =========================================================
# HEAD
# =========================================================
class NightCoxHeadMamba(nn.Module):
    """JOINT head: log_hazard = MLP([ psg_embedding || demographics ]).

    Demographics (age, sex) enter as FEATURES into a small encoder, are
    concatenated with the pooled PSG night embedding, and a joint MLP produces
    all task hazards. This lets the model learn demographic x PSG interactions
    (the additive two-tower split is gone).

    Because the additive decomposition no longer exists, the PSG-vs-demographic
    contribution is measured at EVAL time by the feature-zeroing ablations:
        full      = real PSG features + real demographics
        age_only  = PSG features zeroed -> demographics-only prediction
        full - age_only  ~= PSG-attributable hazard (approx; MLP is nonlinear)
    The head therefore no longer returns psg_tasks/age_tasks.
    """
    def __init__(self, embed_dim: int = 128, mamba_dim: int = 192,
                 n_mamba_layers: int = 2, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, task_names: list = None,
                 groups: dict = None, expected_mods: list = None,
                 demo_dim: int = 48, age_min: float = 18.0, age_max: float = 99.0):
        super().__init__()
        assert task_names is not None and expected_mods is not None
        self.task_names = task_names
        self.expected_mods = expected_mods
        self.mamba_dim = mamba_dim
        self.n_tasks = len(task_names)
        self.age_min, self.age_max = age_min, age_max
        self.groups = groups or {}
        self.group_names = list(self.groups.keys())
        self.n_groups = len(self.group_names)

        task_to_idx = {t: i for i, t in enumerate(task_names)}
        self.group_task_indices = [
            [task_to_idx[t] for t in tasks if t in task_to_idx]
            for tasks in self.groups.values()
        ]

        # ---- PSG path (TRANSFERS from the 3600 checkpoint) ----
        self.mod_in = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(2 * embed_dim, mamba_dim), nn.SiLU(), nn.LayerNorm(mamba_dim),
            ) for mod in expected_mods
        })
        self.mod_fusion = nn.Sequential(
            nn.Linear(len(expected_mods) * mamba_dim, mamba_dim),
            nn.SiLU(), nn.LayerNorm(mamba_dim),
        )
        self.temporal_encoder = BiMambaEncoder(
            d_model=mamba_dim, num_layers=n_mamba_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, share_ffn=True, merge="concat",
        )
        self.pool_proj = nn.Sequential(nn.Linear(mamba_dim * 3, mamba_dim), nn.SiLU())
        self.psg_mlp = nn.Sequential(
            nn.Linear(mamba_dim, mamba_dim * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(mamba_dim * 2, mamba_dim), nn.SiLU(),
        )

        # ---- demographic feature encoder (NEW) ----
        # age normalized to [0,1], sex in {0,1} -> small MLP -> LayerNorm
        self.demo_encoder = nn.Sequential(
            nn.Linear(2, demo_dim), nn.SiLU(), nn.LayerNorm(demo_dim),
        )

        # ---- JOINT head (NEW) ----
        joint_dim = mamba_dim + demo_dim
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_dim, mamba_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.LayerNorm(mamba_dim),
        )
        self.task_head = nn.Linear(mamba_dim, self.n_tasks)
        # keep task logits centered: no per-task constant offset to park a bias in
        nn.init.zeros_(self.task_head.bias)

        if self.n_groups > 0:
            self.group_scale = nn.Embedding(self.n_groups, mamba_dim)
            nn.init.normal_(self.group_scale.weight, mean=0.0, std=0.01)
            self.group_heads = nn.ModuleDict({
                name: nn.Linear(mamba_dim, 1) for name in self.group_names
            })

    # ---------------------------------------------------------------
    def _demo_features(self, age, sex, B, device):
        if age is None:
            age = torch.full((B,), 50.0, device=device)
        if sex is None:
            sex = torch.zeros(B, device=device)
        age = age.float().view(B)
        sex = sex.float().view(B)
        age_n = ((age - self.age_min) / (self.age_max - self.age_min)).clamp(0.0, 1.0)
        return torch.stack([age_n, sex], dim=-1)            # [B, 2]

    def _pool(self, x, temporal_mask=None):
        eps = 1e-6
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if temporal_mask is not None:
            mask = temporal_mask.unsqueeze(-1).float()
            count = mask.sum(1).clamp(min=1.0)
            mean_p = (x * mask).sum(1) / count
            centered = (x - mean_p.unsqueeze(1)) * mask
            std_p = torch.sqrt((centered ** 2).sum(1) / count + eps)
            neg = x.masked_fill(~temporal_mask.unsqueeze(-1), -1e9)
            max_p = neg.max(1).values
            max_p = torch.where(max_p <= -1e8, torch.zeros_like(max_p), max_p)
        else:
            mean_p = x.mean(1)
            std_p = torch.sqrt(x.var(1, unbiased=False) + eps)
            max_p = x.max(1).values
        pooled = torch.cat([mean_p, std_p, max_p], dim=-1)
        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------
    def forward(self, feats_private, feats_shared, expected_mods,
                temporal_mask=None, age=None, sex=None):
        ref = next(iter(feats_shared.values()))
        B, S, _ = ref.shape
        device = ref.device

        tokens = []
        for mod in expected_mods:
            if mod in feats_private and mod in feats_shared:
                pv = torch.nan_to_num(feats_private[mod], 0.0, 0.0, 0.0)
                sh = torch.nan_to_num(feats_shared[mod], 0.0, 0.0, 0.0)
                tokens.append(self.mod_in[mod](torch.cat([pv, sh], dim=-1)))
            else:
                tokens.append(torch.zeros(B, S, self.mamba_dim, device=device))

        x = self.mod_fusion(torch.cat(tokens, dim=-1))
        if temporal_mask is not None:
            x = x * temporal_mask.unsqueeze(-1).to(x.dtype)
        x = self.temporal_encoder(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        pooled = self.pool_proj(self._pool(x, temporal_mask))     # PSG night embedding
        self._last_pooled = pooled.detach()                        # for UMAP
        h_psg = self.psg_mlp(pooled)

        demo = self._demo_features(age, sex, B, device)            # [B, 2]
        demo_emb = self.demo_encoder(demo)                         # [B, demo_dim]

        h = self.joint_mlp(torch.cat([h_psg, demo_emb], dim=-1))   # joint hidden
        task_logits = self.task_head(h)

        if self.n_groups > 0:
            group_logits = torch.cat([
                self.group_heads[name](h * (1.0 + self.group_scale.weight[g]))
                for g, name in enumerate(self.group_names)
            ], dim=1)
        else:
            group_logits = torch.empty(B, 0, device=device)

        return {
            "tasks":  torch.nan_to_num(task_logits, 0.0, 0.0, 0.0),
            "groups": torch.nan_to_num(group_logits, 0.0, 0.0, 0.0),
        }


# =========================================================
# LIGHTNING MODULE
# =========================================================
class CoxPHDownstreamLightning(pl.LightningModule):
    def __init__(
        self,
        backbone_checkpoint_path: str = None,
        embed_dim: int = 128,
        lr_head: float = 1e-3,
        lr_backbone: float = 5e-6,
        weight_decay: float = 1e-4,
        expected_mods: List[str] = None,
        use_lora: bool = True,                 # <-- toggle LoRA on/off
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        # head
        head_mamba_dim: int = 192,
        head_n_mamba_layers: int = 2,
        head_d_state: int = 32,
        head_d_conv: int = 4,
        head_expand: int = 2,
        head_dropout: float = 0.1,
        head_demo_dim: int = 48,               # <-- demographic encoder width
        # multitask
        task_names: List[str] = None,
        groups: Dict[str, List[str]] = None,
        group_loss_weight: float = 0.2,
        grad_clip_norm: float = 1.0,           # <-- clipping that always fires (see hook)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.expected_mods = expected_mods or ["brain", "respiratory", "spo2", "emg", "ecg"]
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.groups = groups or GROUPS
        self.n_groups = len(self.groups)
        self.group_loss_weight = group_loss_weight
        self.grad_clip_norm = grad_clip_norm

        # ---------- Backbone (instantiated EXACTLY as pretrained) ----------
        fm = SleepFoundationModel(
            embed_dim=128, num_heads=4, pooling_heads=8, dropout=0.1,
            base_channels=64, channel_mults=(2, 4, 4), mod_dropout_p=0.2,
            loo_pool_chunks=60, loo_cl_tau=0.07, pair_cl_tau=0.10, pair_offset=6,
            lambda_private=1.0, lambda_loo=1.0, lambda_ortho=0.5,
            lr=1e-4, weight_decay=1e-5,
        )
        if backbone_checkpoint_path:
            print(f"Loading backbone from {backbone_checkpoint_path}")
            ckpt = torch.load(backbone_checkpoint_path, map_location="cpu")
            fm.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        self.backbone = fm.encoder
        del fm

        self.backbone.mod_dropout_p = 0.0
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ---------- LoRA (optional) ----------
        if use_lora:
            apply_lora_to_backbone(self.backbone, rank=lora_rank, alpha=lora_alpha)
            # gradient checkpointing only matters when the backbone has trainable params
            from torch.utils.checkpoint import checkpoint
            original_backbone_forward = self.backbone.forward

            def checkpointed_backbone(modality_groups, channel_masks):
                return checkpoint(original_backbone_forward, modality_groups,
                                  channel_masks, use_reentrant=False)
            self.backbone.forward = checkpointed_backbone
        else:
            print("LoRA disabled — backbone fully frozen (linear-probe style). "
                  "Backbone features are the pretrained ones; only the head trains.")

        # ---------- Head (JOINT: demographics as features) ----------
        self.head = NightCoxHeadMamba(
            embed_dim=embed_dim, mamba_dim=head_mamba_dim,
            n_mamba_layers=head_n_mamba_layers, d_state=head_d_state,
            d_conv=head_d_conv, expand=head_expand, dropout=head_dropout,
            task_names=task_names, groups=self.groups,
            expected_mods=self.expected_mods, demo_dim=head_demo_dim,
        )

        lora_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"Trainable — backbone(LoRA): {lora_params:,} | Head: {head_params:,} | "
              f"Total: {lora_params + head_params:,}")

        self.val_preds = []
        self.val_labels = []

    # Keep frozen backbone in eval mode even when the module trains.
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    # ---------- losses ----------
    def _task_cox_loss(self, logits, events, times):
        total_loss, n_valid = 0.0, 0
        for t in range(self.n_tasks):
            e = events[:, t].bool()
            if e.sum() == 0:
                continue
            total_loss += neg_partial_log_likelihood(logits[:, t], e, times[:, t])
            n_valid += 1
        if n_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return total_loss / n_valid

    def _group_cox_loss(self, group_logits, events, times):
        if self.n_groups == 0 or group_logits.shape[1] == 0:
            return torch.tensor(0.0, device=group_logits.device, requires_grad=True)
        total_loss, n_valid = 0.0, 0
        for g_idx, task_indices in enumerate(self.head.group_task_indices):
            if len(task_indices) == 0:
                continue
            g_events = events[:, task_indices].max(dim=1).values.bool()
            if g_events.sum() == 0:
                continue
            group_times = times[:, task_indices].clone()
            group_times[events[:, task_indices] == 0] = 1e9
            g_times = group_times.min(dim=1).values
            total_loss += neg_partial_log_likelihood(group_logits[:, g_idx], g_events, g_times)
            n_valid += 1
        if n_valid == 0:
            return torch.tensor(0.0, device=group_logits.device, requires_grad=True)
        return total_loss / n_valid

    # ---------- forward ----------
    def backbone_features(self, modality_groups, channel_masks, chunk_size=128):
        ref_mod = next(iter(modality_groups))
        B, C, S, L = modality_groups[ref_mod].shape
        priv_chunks = defaultdict(list)
        shar_chunks = defaultdict(list)
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            chunk_groups = {mod: modality_groups[mod][:, :, start:end, :]
                            for mod in modality_groups}
            out = self.backbone(chunk_groups, channel_masks)
            for mod, val in out["private_raw"].items():
                priv_chunks[mod].append(val)
            for mod, val in out["shared_raw"].items():
                shar_chunks[mod].append(val)
        feats_private = {m: torch.cat(c, dim=1) for m, c in priv_chunks.items()}
        feats_shared = {m: torch.cat(c, dim=1) for m, c in shar_chunks.items()}
        return feats_private, feats_shared

    def head_from_features(self, feats_private, feats_shared,
                           temporal_mask=None, age=None, sex=None):
        return self.head(
            feats_private=feats_private, feats_shared=feats_shared,
            expected_mods=self.expected_mods,
            temporal_mask=temporal_mask, age=age, sex=sex,
        )

    def forward(self, modality_groups, channel_masks, temporal_mask=None,
                age=None, sex=None, chunk_size=128):
        fp, fs = self.backbone_features(modality_groups, channel_masks, chunk_size)
        return self.head_from_features(fp, fs, temporal_mask=temporal_mask, age=age, sex=sex)

    # ---------- train / val ----------
    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        # backbone ONCE; head for the real pass
        fp, fs = self.backbone_features(batch["modality_groups"], batch["channel_masks"])
        out = self.head_from_features(fp, fs, temporal_mask=batch.get("temporal_mask"),
                                      age=batch.get("age"), sex=batch.get("sex"))
        events = batch["events"].float()
        times = batch["times"].float()
        task_loss = self._task_cox_loss(out["tasks"], events, times)
        group_loss = self._group_cox_loss(out["groups"], events, times)
        loss = task_loss + self.group_loss_weight * group_loss

        # skip a poisoned batch instead of letting it NaN the weights
        if not torch.isfinite(loss):
            self.log("train/skipped_nonfinite", 1.0, on_step=True, sync_dist=True)
            return None

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/task_loss", task_loss, prog_bar = True, sync_dist=True)
        self.log("train/group_loss", group_loss, prog_bar = True, sync_dist=True)

        # ---- PSG-vs-demographic diagnostic (joint-head version) ----
        # demo-only logits = head with PSG features zeroed; PSG contribution = full - demo
        
        return loss

    # clip + scrub non-finite grads BEFORE the optimizer step. This hook fires
    # under automatic optimization regardless of the optimizer (AdamMuon included),
    # which Trainer(gradient_clip_val=...) may NOT for a custom optimizer.



    # ---------- validation (held-out set; bf16-safe) ----------
    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        with torch.no_grad():
            out = self.forward(batch["modality_groups"], batch["channel_masks"],
                               temporal_mask=batch.get("temporal_mask"),
                               age=batch.get("age"), sex=batch.get("sex"))
            events = batch["events"].float()
            times = batch["times"].float()
            task_loss = self._task_cox_loss(out["tasks"], events, times)
            group_loss = self._group_cox_loss(out["groups"], events, times)
            val_loss = task_loss + self.group_loss_weight * group_loss

        # only log a finite loss (a degenerate val batch shouldn't poison the monitor)
        if torch.isfinite(val_loss):
            self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # cast to float32 on the way to the CPU buffers: bf16/fp16 -> numpy fails later
        self.val_preds.append(torch.sigmoid(out["tasks"]).float().cpu())
        self.val_labels.append(batch["events"].float().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        preds = torch.cat(self.val_preds).float().numpy()      # .float() = bf16-safe
        labels = torch.cat(self.val_labels).float().numpy()
        self.val_preds.clear()
        self.val_labels.clear()

        aurocs, auprcs = [], []
        for t in range(self.n_tasks):
            yt = labels[:, t]
            if len(np.unique(yt)) < 2:        # task with one class in val -> skip
                continue
            pt = preds[:, t]
            if not np.isfinite(pt).all():     # guard against any stray non-finite preds
                continue
            try:
                aurocs.append(roc_auc_score(yt, pt))
                auprcs.append(average_precision_score(yt, pt))
            except Exception:
                pass
        if aurocs:
            self.log("val/auroc", float(np.mean(aurocs)), prog_bar=True, sync_dist=True)
            self.log("val/auprc", float(np.mean(auprcs)), sync_dist=True)

    # ---------- optimizer ----------
    def configure_optimizers(self):
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        param_groups = [{"params": head_params, "lr": self.hparams.lr_head}]
        if backbone_params:        # empty when use_lora=False
            param_groups.append({"params": backbone_params, "lr": self.hparams.lr_backbone})
        return AdamMuon(param_groups, weight_decay=self.hparams.weight_decay)