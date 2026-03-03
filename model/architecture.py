"""
  ItemTower
      Wraps a pretrained multilingual sentence encoder
      (default: intfloat/multilingual-e5-base).
      Encodes anime text → 128-dim unit-norm embedding.
      The encoder can be fully frozen, LoRA-adapted, or fully trainable.

  UserTower
      Takes a variable-length set of (item_embedding, score_norm) pairs
      → score-conditioned multi-head self-attention
      → mean pool over valid positions
      → 128-dim unit-norm user embedding.

  TwoTowerModel
      Wraps both towers.  During training computes in-batch
      sampled-softmax loss.  At inference time the item tower
      is called once to build a FAISS-ready embedding matrix;
      the user tower is called per-request.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        # Freeze the original weight
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

        # Initialise A with kaiming, B with zeros (so ΔW starts at 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.drop(x))) * self.scale
        return base + lora


def apply_lora_to_encoder(model: nn.Module, rank: int, alpha: float, dropout: float):
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            k in name for k in ("query", "value", "q_proj", "v_proj")
        ):
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr, lora_layer)
            replaced += 1

    return replaced


class ItemTower(nn.Module):
    def __init__(
        self,
        encoder_name: str = "intfloat/multilingual-e5-base",
        proj_dim: int = 128,
        dropout: float = 0.1,
        freeze_mode: str = "lora",  # "all" | "lora" | "none"
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        pooling: str = "mean",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.pooling = pooling

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_mode == "all":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        elif freeze_mode == "lora":
            n = apply_lora_to_encoder(self.encoder, rank=lora_rank,
                                      alpha=lora_alpha, dropout=lora_dropout)
            print(f"[ItemTower] Applied LoRA to {n} linear layers")

        if gradient_checkpointing:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
                print("[ItemTower] Gradient checkpointing enabled")
            else:
                print("[ItemTower] Warning: model does not support gradient_checkpointing_enable()")

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def _pool(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        proj = self.proj(pooled)
        return F.normalize(proj, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_tokens(input_ids, attention_mask)


class UserTower(nn.Module):
    def __init__(
        self,
        proj_dim: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert proj_dim % nhead == 0, "proj_dim must be divisible by nhead"

        self.score_proj = nn.Linear(1, proj_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(proj_dim)

        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(
        self,
        item_embeddings: torch.Tensor,
        context_scores: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = item_embeddings.shape
        score_emb = self.score_proj(context_scores.unsqueeze(-1))
        ctx = item_embeddings + score_emb

        key_padding_mask = ~context_mask
        attn_out, _ = self.attn(
            ctx, ctx, ctx,
            key_padding_mask=key_padding_mask,
        )
        ctx = self.attn_norm(ctx + attn_out)

        valid_float   = context_mask.float().unsqueeze(-1)
        user_vec = (ctx * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp(min=1)

        user_vec = self.out_proj(user_vec)
        return F.normalize(user_vec, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "intfloat/multilingual-e5-base",
        proj_dim: int = 128,
        nhead: int = 4,
        temperature: float = 0.07,
        dropout: float = 0.1,
        freeze_mode: str = "lora",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        pooling: str = "mean",
        gradient_checkpointing: bool = True,
        n_items: int = 0,
        n_users: int = 0,
        cf_dim: int = 0,
    ):
        super().__init__()
        self.temperature = temperature
        self.has_cf = n_items > 0 and cf_dim > 0

        self.item_tower = ItemTower(
            encoder_name=encoder_name,
            proj_dim=proj_dim,
            dropout=dropout,
            freeze_mode=freeze_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pooling=pooling,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.user_tower = UserTower(
            proj_dim=proj_dim,
            nhead=nhead,
            dropout=dropout,
        )

        if self.has_cf:
            self.item_cf_emb = nn.Embedding(n_items, proj_dim, padding_idx=0)
            nn.init.normal_(self.item_cf_emb.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                self.item_cf_emb.weight[0].zero_()

            self.user_cf_emb = nn.Embedding(n_users, proj_dim, padding_idx=0)
            nn.init.normal_(self.user_cf_emb.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                self.user_cf_emb.weight[0].zero_()

            self.cf_gate = nn.Parameter(torch.tensor(-2.2))  # sigmoid(-2.2) ≈ 0.1

    def fuse_item_cf(
        self,
        text_embs: torch.Tensor,
        item_idxs: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_cf:
            return text_embs
        cf = self.item_cf_emb(item_idxs)
        gate = torch.sigmoid(self.cf_gate)
        fused = text_embs + gate * cf
        return F.normalize(fused, p=2, dim=-1)

    def fuse_user_cf(
        self,
        user_embs: torch.Tensor,
        user_idxs: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_cf:
            return user_embs
        cf = self.user_cf_emb(user_idxs)
        gate = torch.sigmoid(self.cf_gate)
        fused = user_embs + gate * cf
        return F.normalize(fused, p=2, dim=-1)

    def encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(input_ids, attention_mask)

    def encode_user(
        self,
        item_embeddings: torch.Tensor,
        context_scores:  torch.Tensor,
        context_mask:    torch.Tensor,
        user_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        user_embs = self.user_tower(item_embeddings, context_scores, context_mask)
        if user_idx is not None and self.has_cf:
            user_embs = self.fuse_user_cf(user_embs, user_idx)
        return user_embs

    def forward(
        self,
        target_input_ids: torch.Tensor,
        target_attn_mask: torch.Tensor,
        context_item_embs: torch.Tensor,
        context_scores: torch.Tensor,
        context_mask: torch.Tensor,
        target_item_idxs: Optional[torch.Tensor] = None,
        user_idxs: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        target_embs = self.item_tower(target_input_ids, target_attn_mask)
        if target_item_idxs is not None and self.has_cf:
            target_embs = self.fuse_item_cf(target_embs, target_item_idxs)

        user_embs = self.user_tower(context_item_embs, context_scores, context_mask)
        if user_idxs is not None and self.has_cf:
            user_embs = self.fuse_user_cf(user_embs, user_idxs)

        logits = torch.matmul(user_embs, target_embs.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss   = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def encode_item_catalog(
        self,
        tokenizer,
        texts: list[str],
        batch_size: int = 256,
        device: str = "cuda",
        max_length: int = 512,
    ) -> torch.Tensor:
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            embs = self.item_tower(**enc)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "intfloat/multilingual-e5-base",
        proj_dim: int = 128,
        dropout: float = 0.1,
        freeze_mode: str = "lora",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        base_margin: float = 0.3,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.base_margin = base_margin
        self.tower = ItemTower(
            encoder_name=encoder_name,
            proj_dim=proj_dim,
            dropout=dropout,
            freeze_mode=freeze_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.tower(input_ids, attention_mask)

    def forward(
        self,
        anchor_ids: torch.Tensor, anchor_mask:   torch.Tensor,
        positive_ids: torch.Tensor, positive_mask: torch.Tensor,
        negative_ids: torch.Tensor, negative_mask: torch.Tensor,
        score_gap: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        a = self._encode(anchor_ids,   anchor_mask)
        p = self._encode(positive_ids, positive_mask)
        n = self._encode(negative_ids, negative_mask)

        d_ap = 1 - (a * p).sum(dim=-1)
        d_an = 1 - (a * n).sum(dim=-1)
        margin = self.base_margin * score_gap.to(a.device)
        
        loss = F.relu(d_ap - d_an + margin).mean()
        return {"loss": loss, "d_ap": d_ap.mean(), "d_an": d_an.mean()}
