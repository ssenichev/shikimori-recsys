"""
The two-tower model retrieves the top-K candidates efficiently (~10ms)
using approximate nearest-neighbour search.  The reranker then takes
those K candidates and re-scores each one by running a single transformer
forward pass that sees BOTH the user profile and the anime description
concatenated as one sequence.

This is much more expressive than the dot-product of two independent
embeddings — the attention mechanism can model fine-grained interactions
like "user likes mecha but only when it has psychological themes".

Architecture
────────────
  Input:  "[CLS] <user_profile_text> [SEP] <anime_text> [SEP]"
  Encoder: DistilBERT (66M params, ~2× faster than BERT-base, good for T4)
  Head:    Linear(hidden, 1)  →  relevance score (trained as regression on score_norm)

  At inference:
    1. Two-tower retrieves top-100 candidates (~10ms)
    2. For each candidate, build the cross-encoder input string
    3. Run in one batched forward pass (~150-200ms on T4 for 100 candidates)
    4. Return candidates sorted by reranker score

User profile text construction
───────────────────────────────
  Rather than encoding the raw interaction history, we build a short
  natural-language summary of the user's taste:
    "User enjoys: Attack on Titan (10/10), Steins;Gate (9/10), ...
     User disliked: [anime] (3/10), ..."

  This is compact, multilingual-friendly, and gives the cross-encoder
  enough signal to understand preference direction without hitting the
  512-token limit.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler as _GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

PROFILE_TOP_LIKED = 8
PROFILE_TOP_DISLIKED = 3
SCORE_HIGH = 7
SCORE_LOW = 4


def build_user_profile_text(
    user_interactions: pd.DataFrame,
    id_to_name: dict[int, str],
    max_liked: int = PROFILE_TOP_LIKED,
    max_disliked: int = PROFILE_TOP_DISLIKED,
) -> str:
    df = user_interactions.copy()

    liked = (
        df[df["is_explicit"] & (df["score_raw"] >= SCORE_HIGH)]
        .sort_values(["score_raw", "confidence"], ascending=False)
        .head(max_liked)
    )
    
    disliked = (
        df[df["is_explicit"] & (df["score_raw"] <= SCORE_LOW)]
        .sort_values("score_raw", ascending=True)
        .head(max_disliked)
    )
    
    implicit = df[~df["is_explicit"]].head(5)
    parts = []

    if not liked.empty:
        entries = []
        for _, row in liked.iterrows():
            name = id_to_name.get(int(row["anime_id"]), f"[{int(row['anime_id'])}]")
            entries.append(f"{name} ({int(row['score_raw'])})")
        parts.append("Enjoyed: " + ", ".join(entries))

    if not implicit.empty:
        names = [id_to_name.get(int(r["anime_id"]), f"[{int(r['anime_id'])}]")
                 for _, r in implicit.iterrows()]
        parts.append("Also watched: " + ", ".join(names))

    if not disliked.empty:
        entries = []
        for _, row in disliked.iterrows():
            name = id_to_name.get(int(row["anime_id"]), f"[{int(row['anime_id'])}]")
            entries.append(f"{name} ({int(row['score_raw'])})")
        parts.append("Disliked: " + ", ".join(entries))

    if not parts:
        return "No watch history available."

    return " | ".join(parts)

class CrossEncoderReranker(nn.Module):
    def __init__(
        self,
        encoder_name: str = "distilbert-base-multilingual-cased",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls).squeeze(-1)
        return torch.sigmoid(logits)

class RerankerDataset(Dataset):
    def __init__(
        self,
        train_df: pd.DataFrame,
        anime_df: pd.DataFrame,
        id_to_name: dict[int, str],
        negatives_per_user: int = 3,
        max_liked: int = PROFILE_TOP_LIKED,
        e5_prefix: str = "passage: ",
    ):
        self.e5_prefix = e5_prefix

        id_to_text: dict[int, str] = {
            int(row["id"]): str(row["text_input"])
            for _, row in anime_df.iterrows()
        }
        all_anime_ids = list(id_to_text.keys())

        self.samples: list[dict] = []

        for user_id, group in train_df.groupby("user_id"):
            if "created_at" in group.columns:
                group = group.sort_values("created_at")
            else:
                group = group.sort_values("confidence", ascending=False)

            user_anime_ids = set(group["anime_id"].astype(int).tolist())

            explicit = group[group["is_explicit"]] if "is_explicit" in group.columns else group

            for _, row in explicit.iterrows():
                target_id = int(row["anime_id"])
                score_norm  = float(row["score_norm"])

                if "created_at" in group.columns:
                    ctx = group[
                        (group["anime_id"] != target_id) &
                        (group["created_at"] <= row["created_at"])
                    ]
                else:
                    ctx = group[group["anime_id"] != target_id]

                if ctx.empty:
                    continue

                profile_text = build_user_profile_text(
                    ctx, id_to_name, max_liked=max_liked
                )
                anime_text = id_to_text.get(target_id, "")
                if not anime_text:
                    continue

                self.samples.append({
                    "profile_text": profile_text,
                    "anime_text": e5_prefix + anime_text,
                    "score_norm": score_norm,
                })

            if negatives_per_user > 0:
                full_profile = build_user_profile_text(group, id_to_name, max_liked=max_liked)
                unseen = [a for a in all_anime_ids if a not in user_anime_ids]
                neg_ids = np.random.choice(
                    unseen, size=min(negatives_per_user, len(unseen)), replace=False
                )
                for neg_id in neg_ids:
                    anime_text = id_to_text.get(int(neg_id), "")
                    if not anime_text:
                        continue
                    self.samples.append({
                        "profile_text": full_profile,
                        "anime_text": e5_prefix + anime_text,
                        "score_norm": 0.0,
                    })

        import random
        random.shuffle(self.samples)
        log.info(
            "RerankerDataset: %d samples (%d users)",
            len(self.samples), train_df["user_id"].nunique(),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "profile_text": s["profile_text"],
            "anime_text": s["anime_text"],
            "score_norm": torch.tensor(s["score_norm"], dtype=torch.float),
        }


def collate_reranker(batch: list[dict], tokenizer, max_length: int = 256) -> dict:
    profiles = [s["profile_text"] for s in batch]
    animes = [s["anime_text"]   for s in batch]
    targets = torch.stack([s["score_norm"] for s in batch])

    enc = tokenizer(
        profiles, animes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["targets"] = targets
    return enc


def train_stage3(
    reranker: CrossEncoderReranker,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> CrossEncoderReranker:
    log.info("=" * 60)
    log.info("Stage 3: Cross-encoder reranker training")
    log.info("=" * 60)

    id_to_name: dict[int, str] = {
        int(row["id"]): str(row.get("name", row.get("text_input", "")))
        for _, row in anime_df.iterrows()
    }

    train_ds = RerankerDataset(
        train_df, anime_df, id_to_name,
        negatives_per_user=cfg.get("s3_neg_per_user", 3),
    )

    _collate = lambda batch: collate_reranker(batch, tokenizer, cfg["s3_max_length"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["s3_batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    optimizer = AdamW(
        reranker.parameters(),
        lr=cfg["s3_lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps  = len(train_loader) * cfg["s3_epochs"]
    warmup_steps = cfg.get("s3_warmup_steps", max(1, total_steps // 10))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )

    scaler = _GradScaler(enabled=(device.type == "cuda"))
    grad_accum = cfg.get("s3_grad_accum", 2)
    reranker.to(device)

    best_loss = float("inf")

    for epoch in range(1, cfg["s3_epochs"] + 1):
        reranker.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                preds = reranker(input_ids, attention_mask)   # [B]
                loss  = F.mse_loss(preds, targets) / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    reranker.parameters(), cfg.get("s3_grad_clip", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum

            if step % 100 == 0:
                log.info(
                    "S3 E%d/%d  step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, cfg["s3_epochs"], step, len(train_loader),
                    loss.item() * grad_accum,
                    scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / len(train_loader)
        log.info("S3 Epoch %d done  avg_loss=%.4f", epoch, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(reranker.state_dict(), output_dir / "stage3_best.pt")
            log.info("  ✓ Saved Stage 3 best checkpoint (loss=%.4f)", best_loss)

    reranker.load_state_dict(
        torch.load(output_dir / "stage3_best.pt", map_location=device)
    )
    log.info("Stage 3 complete. Best MSE loss: %.4f", best_loss)
    return reranker


def _history_list_to_df(
    history: "list[tuple[int, int]]",
) -> pd.DataFrame:
    rows = []
    for anime_id, score in history:
        is_explicit  = (score > 0)
        score_norm   = (score - 1) / 9.0 if is_explicit else 0.75
        confidence   = score_norm
        rows.append({
            "anime_id":    int(anime_id),
            "score_raw":   int(score),
            "score_norm":  score_norm,
            "confidence":  confidence,
            "is_explicit": is_explicit,
            "rewatches":   0,
        })
    return pd.DataFrame(rows)


class TwoTowerWithReranker:
    def __init__(
        self,
        two_tower: "TwoTowerModel",   # noqa: F821
        reranker: CrossEncoderReranker,
        tokenizer, # two-tower tokenizer (e5 / XLM-R)
        anime_df: pd.DataFrame,
        device: torch.device,
        reranker_tokenizer  = None,
        max_length: int = 128,
        rerank_max_length: int = 256,
        item_id_to_cf_idx: Optional[dict[int, int]] = None,
        user_id_to_cf_idx: Optional[dict[int, int]] = None,
    ):
        self.two_tower = two_tower
        self.reranker  = reranker
        self.tokenizer = tokenizer
        self.reranker_tokenizer = reranker_tokenizer if reranker_tokenizer is not None else tokenizer

        if reranker_tokenizer is None:
            log.warning(
                "TwoTowerWithReranker: reranker_tokenizer not provided, "
                "falling back to two-tower tokenizer. This will cause errors "
                "if the two-tower and reranker use different base models."
            )
        self.device    = device
        self.max_length       = max_length
        self.rerank_max_length = rerank_max_length

        self.anime_df  = anime_df.copy()
        self.id_to_idx: dict[int, int] = {}
        self.idx_to_id: dict[int, int] = {}
        self.id_to_text: dict[int, str] = {}
        self.id_to_name: dict[int, str] = {}
        self.item_matrix: Optional[torch.Tensor] = None

        # CF mappings
        self.item_id_to_cf_idx = item_id_to_cf_idx or {}
        self.user_id_to_cf_idx = user_id_to_cf_idx or {}

        self._build_catalogue()

    def _build_catalogue(self):
        log.info("Building item embedding catalogue...")
        texts, ids = [], []
        for _, row in self.anime_df.iterrows():
            aid = int(row["id"])
            self.id_to_text[aid] = str(row["text_input"])
            self.id_to_name[aid] = str(row.get("name", ""))
            ids.append(aid)
            texts.append("passage: " + str(row["text_input"]))

        self.id_to_idx = {aid: i for i, aid in enumerate(ids)}
        self.idx_to_id = {i: aid for i, aid in enumerate(ids)}

        self.two_tower.eval()
        all_embs = []
        bs = 128
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tokenizer(
                    texts[i:i+bs],
                    padding=True, truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    embs = self.two_tower.item_tower(**enc)
                all_embs.append(embs.cpu())

        self.item_matrix = torch.cat(all_embs, dim=0)

        # Fuse CF embeddings if available
        if self.two_tower.has_cf and self.item_id_to_cf_idx:
            cf_idxs = torch.tensor(
                [self.item_id_to_cf_idx.get(aid, 0) for aid in ids],
                dtype=torch.long,
            )
            with torch.no_grad():
                self.item_matrix = self.two_tower.fuse_item_cf(
                    self.item_matrix.to(self.device), cf_idxs.to(self.device)
                ).cpu()

        log.info("Catalogue ready: %d items × %d dims", len(ids), self.item_matrix.shape[1])

    @torch.no_grad()
    def _encode_user(
        self,
        user_history: pd.DataFrame,
        max_history: int = 50,
        user_id: Optional[int] = None,
    ) -> torch.Tensor:
        if "created_at" in user_history.columns:
            history = user_history.sort_values("created_at").tail(max_history)
        else:
            sort_col = "confidence" if "confidence" in user_history.columns else "score_norm"
            history = user_history.sort_values(sort_col, ascending=False).head(max_history)

        context_ids    = history["anime_id"].astype(int).tolist()
        context_scores = history["score_norm"].tolist()

        known = [(aid, s) for aid, s in zip(context_ids, context_scores)
                 if aid in self.id_to_idx]
        if not known:
            D = self.item_matrix.shape[1]
            return torch.zeros(D)
        context_ids, context_scores = zip(*known)

        ctx_idxs = torch.tensor([self.id_to_idx[a] for a in context_ids], dtype=torch.long)
        ctx_embs = self.item_matrix[ctx_idxs].unsqueeze(0).to(self.device)   # [1, L, D]
        ctx_scores= torch.tensor([list(context_scores)], dtype=torch.float).to(self.device)
        ctx_mask = torch.ones(1, len(context_ids), dtype=torch.bool).to(self.device)

        # Build user CF index if available
        user_idx = None
        if self.two_tower.has_cf and user_id is not None and self.user_id_to_cf_idx:
            cf_idx = self.user_id_to_cf_idx.get(int(user_id), 0)
            user_idx = torch.tensor([cf_idx], dtype=torch.long).to(self.device)

        self.two_tower.eval()
        with autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
            user_vec = self.two_tower.encode_user(ctx_embs, ctx_scores, ctx_mask, user_idx=user_idx)
        return user_vec.squeeze(0).cpu()

    @torch.no_grad()
    def _rerank(
        self,
        profile_text:    str,
        candidate_ids:   list[int],
    ) -> list[tuple[int, float]]:
        self.reranker.eval()
        texts = [profile_text] * len(candidate_ids)
        animes = ["passage: " + self.id_to_text.get(aid, "") for aid in candidate_ids]

        enc = self.reranker_tokenizer(
            texts, animes,
            padding=True, truncation=True,
            max_length=self.rerank_max_length,
            return_tensors="pt",
        ).to(self.device)

        with autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
            scores = self.reranker(enc["input_ids"], enc["attention_mask"])

        scored = list(zip(candidate_ids, scores.cpu().tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def recommend(
        self,
        user_history: "list[tuple[int, int]] | pd.DataFrame",
        top_k: int = 10,
        retrieval_k: int = 100,
        exclude_seen: bool = True,
        user_id: Optional[int] = None,
    ) -> list[dict]:
        assert self.item_matrix is not None, "Call _build_catalogue() first"

        if isinstance(user_history, list):
            user_history = _history_list_to_df(user_history)

        user_vec = self._encode_user(user_history, user_id=user_id)
        scores = (user_vec.unsqueeze(0) @ self.item_matrix.T).squeeze(0)

        if exclude_seen:
            seen_ids = set(user_history["anime_id"].astype(int).tolist())
            seen_idxs = [self.id_to_idx[a] for a in seen_ids if a in self.id_to_idx]
            for idx in seen_idxs:
                scores[idx] = -1e9

        top_retrieval_idxs = scores.topk(retrieval_k).indices.tolist()
        candidate_ids = [self.idx_to_id[i] for i in top_retrieval_idxs]

        profile_text = build_user_profile_text(user_history, self.id_to_name)
        reranked = self._rerank(profile_text, candidate_ids)
        id_to_row = self.anime_df.set_index("id")
        results = []
        for rank, (aid, score) in enumerate(reranked[:top_k], 1):
            row = id_to_row.loc[aid] if aid in id_to_row.index else {}
            results.append({
                "rank": rank,
                "anime_id": aid,
                "name": self.id_to_name.get(aid, ""),
                "reranker_score": round(score, 4),
                "two_tower_rank": candidate_ids.index(aid) + 1,
                "genres": list(row["genre_names"]) if "genre_names" in row.index and row["genre_names"] is not None else [],
                "global_score": float(row.get("score_global", 0)) if isinstance(row, dict) else float(row["score_global"]) if "score_global" in row.index else 0.0,
            })

        return results
