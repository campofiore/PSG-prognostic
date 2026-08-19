from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from AdamMuon import AdamMuon


# =============================================================================
# Helpers + ResBlock + Stem + Tokenizer + AttentionPooling  (unchanged)
# =============================================================================
def _make_norm_2d(n: int) -> nn.Module:
    g = min(8, n)
    while n % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, n)


def _make_norm_1d(n: int) -> nn.Module:
    g = max(1, n // 8)
    return nn.GroupNorm(g, n)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = _make_norm_1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = _make_norm_1d(out_ch)
        self.down = None
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                _make_norm_1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(out + identity)


class FreqCollapseResStem(nn.Module):
    """Frequency collapse is intentional: a single tall kernel collapses the
    full spectral height to 1 in one shot, leaving a temporal sequence."""
    def __init__(self, out_ch: int, spec_height: int = 257, k_t: int = 5):
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.conv_main = nn.Conv2d(1, out_ch, (spec_height, k_t), 1, (0, k_t // 2), bias=False)
        self.norm_main = _make_norm_2d(out_ch)
        self.conv_skip = nn.Conv2d(1, out_ch, (spec_height, 1), 1, 0, bias=False)
        self.norm_skip = _make_norm_2d(out_ch)
        self.conv_ref = nn.Conv2d(out_ch, out_ch, (1, 3), 1, (0, 1), bias=False)
        self.norm_ref = _make_norm_2d(out_ch)

    def forward(self, x):
        main = self.norm_main(self.conv_main(x))
        skip = self.norm_skip(self.conv_skip(x))
        y = self.act(main + skip)
        return self.act(self.norm_ref(self.conv_ref(y)) + y)


class SpectrogramEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, base_channels: int = 64,
                 channel_mults: Tuple = (2, 4, 4), stem_k_t: int = 5):
        super().__init__()
        self.base_channels = base_channels
        self.embed_dim = embed_dim
        self.stem = FreqCollapseResStem(self.base_channels, spec_height=257, k_t=stem_k_t)

        stages = []
        in_c = base_channels
        for mult in channel_mults:
            out_c = base_channels * mult
            stages.append(ResBlock1D(in_c, out_c, stride=2))
            stages.append(ResBlock1D(out_c, out_c))
            in_c = out_c
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_c, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BS, C, Fdim, T = x.shape
        x = x.reshape(BS * C, 1, Fdim, T)
        h = self.stem(x).squeeze(2)
        h = self.stages(h)
        h = self.pool(h).squeeze(-1)
        h = self.proj(h)
        return h.view(BS, C, self.embed_dim)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu"
        )

    def forward(self, x, key_padding_mask=None):
        # convention: key_padding_mask True == padded/ignore
        if key_padding_mask is not None:
            if key_padding_mask.size(1) == 1:
                return x.mean(dim=1)
            out = self.layer(x, src_key_padding_mask=key_padding_mask.bool())
            valid = (~key_padding_mask).float().unsqueeze(-1)
            return (out * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.layer(x).mean(dim=1)


# =============================================================================
# Cross-Modal Fusion: modalities-as-tokens + masked self-attention
# =============================================================================
class CrossModalFusion(nn.Module):
    """
    By fusion time each modality is ONE token per (B, S) (channels already
    pooled). The cross-modal "sequence" is therefore just M <= 5 tokens, so a
    single masked self-attention layer over the modality set is the natural fit:

      * variable availability handled natively via key_padding_mask
        (absence == masked key, no train/test distribution shift),
      * one shared parameter set trained on every co-occurrence (no O(M^2) blowup),
      * content-based routing for free (SpO2->EEG can be strong, EEG->SpO2 weak).

    Modality dropout (training only) masks a subset of present modalities AS KEYS.
    A dropped modality still appears as a QUERY, so it gets a valid fused output
    (its own residual + attention over the surviving keys) while contributing
    nothing to the others -- giving the robustness benefit with no extra forward
    pass and no NaNs (we always keep >= 2 keys live).
    """
    def __init__(self, embed_dim: int, modalities: List[str],
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.modalities = modalities
        self.mod_to_idx = {m: i for i, m in enumerate(modalities)}

        # Learned per-modality identity (self-attention is permutation-invariant
        # without it -- the model needs to know which slot is which modality).
        self.mod_embed = nn.Parameter(torch.zeros(len(modalities), embed_dim))
        nn.init.normal_(self.mod_embed, std=0.02)

        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )

    def forward(self, private_tokens: Dict[str, torch.Tensor],
                available: List[str],
                mod_dropout_p: float = 0.2) -> Dict[str, torch.Tensor]:
        if len(available) < 2:
            return {m: private_tokens[m] for m in available}

        ref = private_tokens[available[0]]
        B, S, D = ref.shape
        M = len(self.modalities)
        device = ref.device

        # --- choose which present modalities act as live KEYS (>=2 kept) ---
        keep_as_key = set(available)
        if self.training and len(available) > 2 and mod_dropout_p > 0:
            kept = [m for m in available
                    if torch.rand((), device=device).item() > mod_dropout_p]
            if len(kept) >= 2:
                keep_as_key = set(kept)

        # --- build the M-token stack + key-padding mask (True == ignore) ---
        stack = torch.zeros(B * S, M, D, device=device)
        pad_mask = torch.ones(B * S, M, dtype=torch.bool, device=device)
        for m in available:                      # fill ALL available as queries
            i = self.mod_to_idx[m]
            stack[:, i] = private_tokens[m].reshape(B * S, D) + self.mod_embed[i]
            if m in keep_as_key:                 # only live keys are unmasked
                pad_mask[:, i] = False

        out = self.layer(stack, src_key_padding_mask=pad_mask)  # [B*S, M, D]

        # every AVAILABLE modality gets a shared output (dropped-as-key mods too)
        return {m: out[:, self.mod_to_idx[m]].reshape(B, S, D) for m in available}


# =============================================================================
# Projector (downstream/loss features are the PROJECTED outputs)
# =============================================================================
class Projector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, S, D = x.shape
            return self.net(x.reshape(B * S, D)).reshape(B, S, D)
        return self.net(x)


# =============================================================================
# PSGEncoder
# =============================================================================
class PSGEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        pooling_heads: int = 8,
        dropout: float = 0.0,
        base_channels: int = 64,
        channel_mults: Tuple = (2, 4, 4),
        mod_dropout_p: float = 0.2,
        loo_pool_chunks: int = 360,   # 360 * 5s = 30 min pooling window for LOO-CL
    ):
        super().__init__()
        self.modalities = ["brain", "respiratory", "spo2", "emg", "ecg"]
        self.n_mods = len(self.modalities)
        self.mod_dropout_p = mod_dropout_p
        self.loo_pool_chunks = loo_pool_chunks

        self.enc = SpectrogramEncoder(
            embed_dim=embed_dim,
            base_channels=base_channels,
            channel_mults=channel_mults,
        )

        self.spatial_pool = AttentionPooling(
            embed_dim, num_heads=pooling_heads, dropout=dropout
        )
        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=embed_dim, modalities=self.modalities,
            num_heads=num_heads, dropout=dropout,
        )

        # Projectors: the projected features are what every loss consumes.
        self.private_projector = Projector(embed_dim)
        self.shared_projector = Projector(embed_dim)

        # STFT parameters for 5-second windows
        self.register_buffer("stft_window", torch.hann_window(256, periodic=True), persistent=False)
        self.n_fft = 512
        self.hop_length = 64
        self.win_length = 256

    def _to_spec_5s(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, S, L] raw 5s chunks -> [B, C, S, F, T] log-magnitude spectrograms."""
        B, C, S, L = x.shape
        x_flat = x.reshape(B * C * S, L)

        specs = []
        chunk_size = 2048
        for i in range(0, x_flat.shape[0], chunk_size):
            chunk = x_flat[i:i + chunk_size]
            spec = torch.stft(
                chunk,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.stft_window,
                center=True,
                normalized=False,
                return_complex=True,
            ).abs().log1p()
            specs.append(spec)

        spec_full = torch.cat(specs, dim=0)
        Fdim, T = spec_full.shape[-2], spec_full.shape[-1]
        return spec_full.reshape(B, C, S, Fdim, T)

    def _pool_to_loo_window(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] per-5s -> [B, n_windows, D] pooled over loo_pool_chunks."""
        B, S, D = x.shape
        w = self.loo_pool_chunks
        if S < w:
            return x.mean(dim=1, keepdim=True)
        S_trim = (S // w) * w
        return x[:, :S_trim].reshape(B, S_trim // w, w, D).mean(dim=2)

    def forward(self, modality_groups, channel_masks):
        private_raw: Dict[str, torch.Tensor] = {}   # pre-projector (for fine-tuning)
        available: List[str] = []

        spec_groups = {mod: self._to_spec_5s(x) for mod, x in modality_groups.items()}

        # ---- per-modality encode + channel (spatial) pooling -> private ----
        for mod in self.modalities:
            if mod not in spec_groups:
                continue

            x = spec_groups[mod]                         # [B, C, S, F, T]
            B, C, S, Fdim, T = x.shape

            mask = channel_masks.get(mod)
            if mask is not None:                         # [B, C] -> [B*S, C]
                mask = mask.unsqueeze(1).expand(-1, S, -1).reshape(B * S, C)

            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * S, C, Fdim, T)
            tokens = self.enc(x_flat)                    # [B*S, C, E]

            pooled = self.spatial_pool(
                tokens, (~mask.bool()) if mask is not None else None
            )                                            # [B*S, E]
            private_raw[mod] = pooled.view(B, S, -1)     # [B, S, E]
            available.append(mod)

        # ---- cross-modal fusion (per 5s) -> shared ----
        if len(available) >= 2:
            shared_raw = self.cross_modal_fusion(
                private_raw, available,
                mod_dropout_p=self.mod_dropout_p if self.training else 0.0,
            )
        else:
            shared_raw = {m: private_raw[m] for m in available}

        # ---- project; downstream losses use these projected features ----
        proj_private = {m: self.private_projector(private_raw[m]) for m in available}
        proj_shared = {m: self.shared_projector(shared_raw[m]) for m in available}

        # ---- 30-min pooled SHARED view for LOO-CL ----
        proj_shared_loo = {m: self._pool_to_loo_window(proj_shared[m]) for m in available}

        return {
            "proj_private": proj_private,         # [B, S, D] per mod (projected)
            "proj_shared": proj_shared,           # [B, S, D] per mod (projected)
            "proj_shared_loo": proj_shared_loo,   # [B, n_win, D] per mod (projected)
            "private_raw": private_raw,           # pre-projector, for fine-tuning
            "shared_raw": shared_raw,             # pre-projector, for fine-tuning
            "available_mods": available,
        }


class SleepFoundationModel(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        pooling_heads: int = 8,
        dropout: float = 0.1,
        base_channels: int = 64,
        channel_mults: Tuple = (2, 4, 4),
        mod_dropout_p: float = 0.2,
        loo_pool_chunks: int = 360,
        loo_cl_tau: float = 0.07,
        pair_cl_tau: float = 0.1,
        pair_offset: int = 1,        # temporal positive = chunk s <-> chunk s+offset
        lambda_private: float = 1.0,   # weights the PRIVATE temporal pairwise CL
        lambda_loo: float = 1.0,       # weights the SHARED LOO-CL
        lambda_ortho: float = 0.5,     # weights private<->shared Barlow decorrelation
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = PSGEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooling_heads=pooling_heads,
            dropout=dropout,
            base_channels=base_channels,
            channel_mults=channel_mults,
            mod_dropout_p=mod_dropout_p,
            loo_pool_chunks=loo_pool_chunks,
        )

        self.loo_cl_tau = loo_cl_tau
        self.pair_cl_tau = pair_cl_tau
        self.pair_offset = pair_offset
        self.lambda_private = lambda_private
        self.lambda_loo = lambda_loo
        self.lambda_ortho = lambda_ortho
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, modality_groups, channel_masks):
        return self.encoder(modality_groups, channel_masks)

    # -------------------------------------------------------------------------
    # PRIVATE objective: temporal pairwise CL (no augmentation; time is the view)
    #   positives : same modality, nearby chunks  (b, s) <-> (b, s+offset)
    #   negatives : all other (b', s') of the SAME modality in the batch
    # Within-modality only -> pulls a modality's own chunks together over time,
    # never pulls different modalities together -> no conflict with Barlow.
    # (For the cross-modal flavor instead, pair mod_A[b,s] with mod_B[b,s].)
    # -------------------------------------------------------------------------
    def compute_pairwise_cl(self, proj_private: dict, available: List[str]) -> torch.Tensor:
        if not available:
            return torch.tensor(0.0, device=self.device)
        tau = float(self.pair_cl_tau)
        off = int(self.pair_offset)
        losses = []
        for mod in available:
            z = proj_private[mod]                 # [B, S, D]
            B, S, D = z.shape
            if S <= off:
                continue
            a = F.normalize(z[:, :-off].reshape(-1, D), dim=-1)   # anchors  [N, D]
            p = F.normalize(z[:, off:].reshape(-1, D), dim=-1)    # positives[N, D]
            N = a.shape[0]
            logits = (a @ p.T) / tau                              # [N, N]
            labels = torch.arange(N, device=self.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) +
                          F.cross_entropy(logits.T, labels))
            losses.append(loss)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    # -------------------------------------------------------------------------
    # SHARED objective: leave-one-out contrastive on 30-min pooled shared feats
    #   anchor   : modality m's shared rep
    #   positive : mean of the OTHER modalities' shared reps (same b, window)
    # -------------------------------------------------------------------------
    def compute_loo_cl(self, proj_shared_loo: dict, available: List[str]) -> torch.Tensor:
        if len(available) < 2:
            return torch.tensor(0.0, device=self.device)
        tau = float(self.loo_cl_tau)
        losses = []
        for mod in available:
            others = [m for m in available if m != mod]
            anchor = proj_shared_loo[mod]                 # [B, T, D]
            B, T, D = anchor.shape
            loo_avg = torch.stack([proj_shared_loo[m] for m in others], dim=0).mean(0)

            a = F.normalize(anchor.reshape(B * T, D), dim=-1)
            p = F.normalize(loo_avg.reshape(B * T, D), dim=-1)
            logits = (a @ p.T) / tau
            labels = torch.arange(B * T, device=self.device)
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()

    # -------------------------------------------------------------------------
    # DECORRELATION: separate modality-intrinsic (private) from crossmodal (shared)
    #   primary : per-modality private <-> shared cross-correlation -> 0
    #   bonus   : private_i <-> private_j (i != j) cross-correlation -> 0
    # Operates on the projected, standardized features so the fusion residual
    # (shared ~= private + cross_info) doesn't dominate the penalty.
    # -------------------------------------------------------------------------
    def compute_barlow(self, proj_private: dict, proj_shared: dict,
                       available: List[str], eps: float = 1e-5) -> torch.Tensor:
        if not available:
            return torch.tensor(0.0, device=self.device)

        def standardize(z):
            z = z.reshape(-1, z.shape[-1])
            return (z - z.mean(0)) / (z.std(0) + eps)

        priv = {m: standardize(proj_private[m]) for m in available}
        shar = {m: standardize(proj_shared[m]) for m in available}
        N = priv[available[0]].shape[0]

        losses = []
        # private <-> shared, per modality (the core intrinsic/crossmodal split)
        for m in available:
            losses.append(torch.mm(priv[m].T, shar[m]).div(N).pow(2).mean())
        # private <-> private, across distinct modality pairs (keep subspaces distinct)
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                a, b = priv[available[i]], priv[available[j]]
                losses.append(torch.mm(a.T, b).div(N).pow(2).mean())

        return torch.stack(losses).mean()

    # -------------------------------------------------------------------------
    def compute_private_diagnostics(self, proj_private: dict, proj_shared: dict,
                                    available: list):
        if len(available) < 2:
            return {}
        metrics = {}
        max_tokens = 32
        D = proj_private[available[0]].shape[-1]
        embs, labels = [], []
        for i, mod in enumerate(available):
            z = proj_private[mod]
            B, S, _ = z.shape
            idx = torch.randperm(S, device=z.device)[:max_tokens]
            z_sub = z[:, idx].reshape(-1, D)
            embs.append(z_sub)
            labels.append(torch.full((z_sub.shape[0],), i, device=z.device))

        embs_cat = torch.cat(embs, dim=0)
        labels_cat = torch.cat(labels, dim=0)
        embs_norm = F.normalize(embs_cat, dim=-1)
        n_mods = len(available)

        centroids = F.normalize(torch.stack([
            embs_norm[labels_cat == i].mean(0) for i in range(n_mods)
        ]), dim=-1)
        predicted = torch.matmul(embs_norm, centroids.T).argmax(dim=-1)
        metrics["private/modality_separability"] = (predicted == labels_cat).float().mean()

        sim_matrix = torch.matmul(embs_norm, embs_norm.T)
        same_mask = (labels_cat.unsqueeze(0) == labels_cat.unsqueeze(1))
        eye_mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=embs_cat.device)
        same_mask = same_mask & ~eye_mask
        metrics["private/intra_modal_sim"] = sim_matrix[same_mask].mean()
        metrics["private/inter_modal_sim"] = sim_matrix[~same_mask & ~eye_mask].mean()

        ps_sims = []
        for mod in available:
            priv = F.normalize(proj_private[mod].reshape(-1, D), dim=-1)
            shar = F.normalize(proj_shared[mod].reshape(-1, D), dim=-1)
            idx = torch.randperm(priv.shape[0], device=priv.device)[:256]
            ps_sims.append((priv[idx] * shar[idx]).sum(dim=-1).mean())
        metrics["private/shared_cosine"] = torch.stack(ps_sims).mean()

        return metrics

    # -------------------------------------------------------------------------
    def _shared_step(self, batch, prefix):
        out = self(batch["modality_groups"], batch["channel_masks"])
        available = out["available_mods"]

        loss_pair = self.compute_pairwise_cl(out["proj_private"], available)
        loss_loo = self.compute_loo_cl(out["proj_shared_loo"], available)
        loss_barlow = self.compute_barlow(out["proj_private"], out["proj_shared"], available)

        total = (self.lambda_private * loss_pair +
                 self.lambda_loo * loss_loo +
                 self.lambda_ortho * loss_barlow)

        self.log(f"{prefix}/total", total, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/pairwise_cl", loss_pair, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/loo_cl", loss_loo, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/barlow", loss_barlow, sync_dist=True)

        if prefix == "train" and self.global_step % 50 == 0:
            diag = self.compute_private_diagnostics(
                out["proj_private"], out["proj_shared"], available
            )
            for k, v in diag.items():
                self.log(k, v, prog_bar=True, sync_dist=False)

        if torch.cuda.is_available():
            self.log(f"{prefix}/gpu_gb", torch.cuda.memory_allocated() / 1e9, sync_dist=False)

        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return AdamMuon(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)