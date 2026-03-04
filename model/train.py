import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer

from data.dataset import (
    build_id_to_text,
    make_triplet_loader,
    make_user_ratings_loader,
)
from model.architecture import ContrastiveEncoder, TwoTowerModel
from model.reranker import CrossEncoderReranker, TwoTowerWithReranker, train_stage3
from model.metrics import evaluate_retrieval, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_cf_mappings(
    anime_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], int, int]:
    """Build ID→index mappings for CF embeddings. Index 0 is reserved as padding.

    Returns (item_id_to_idx, user_id_to_idx, n_items, n_users) where
    n_items/n_users are the embedding table sizes (max_index + 1).
    """
    item_id_to_idx: dict[int, int] = {}
    for i, aid in enumerate(anime_df["id"].astype(int).tolist(), start=1):
        item_id_to_idx[aid] = i

    user_id_to_idx: dict[int, int] = {}
    for i, uid in enumerate(sorted(train_df["user_id"].unique()), start=1):
        user_id_to_idx[int(uid)] = i

    n_items = max(item_id_to_idx.values()) + 1 if item_id_to_idx else 1
    n_users = max(user_id_to_idx.values()) + 1 if user_id_to_idx else 1

    return item_id_to_idx, user_id_to_idx, n_items, n_users


def tokenise_batch(
    texts: list[str],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def train_stage1(
    model: ContrastiveEncoder,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> ContrastiveEncoder:
    log.info("=" * 60)
    log.info("Stage 1: Contrastive encoder fine-tuning")
    log.info("=" * 60)

    id_to_text = build_id_to_text(anime_df)
    train_loader = make_triplet_loader(
        train_df,
        id_to_text,
        batch_size=cfg["s1_batch_size"],
        negatives_per_user=cfg["s1_neg_per_user"],
        num_workers=cfg.get("num_workers", 2),
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["s1_lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = len(train_loader) * cfg["s1_epochs"]
    warmup_steps = cfg.get("s1_warmup_steps", max(1, total_steps // 10))

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))
    model.to(device)
    model.train()

    best_val_loss = float("inf")

    for epoch in range(1, cfg["s1_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        grad_accum = cfg.get("s1_grad_accum", 1)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            anchor_enc = tokenise_batch(batch["anchor_text"], tokenizer, cfg["s1_max_length"], device)
            positive_enc = tokenise_batch(batch["positive_text"], tokenizer, cfg["s1_max_length"], device)
            negative_enc = tokenise_batch(batch["negative_text"], tokenizer, cfg["s1_max_length"], device)
            score_gap = batch["score_gap"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                out = model(
                    anchor_ids=anchor_enc["input_ids"],
                    anchor_mask=anchor_enc["attention_mask"],
                    positive_ids=positive_enc["input_ids"],
                    positive_mask=positive_enc["attention_mask"],
                    negative_ids=negative_enc["input_ids"],
                    negative_mask=negative_enc["attention_mask"],
                    score_gap=score_gap,
                )
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("s1_grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum

            if step % 50 == 0:
                log.info(
                    "S1 E%d/%d  step %d/%d  loss=%.4f  d_ap=%.4f  d_an=%.4f  lr=%.2e",
                    epoch, cfg["s1_epochs"], step, len(train_loader),
                    loss.item() * grad_accum, out["d_ap"].item(), out["d_an"].item(),
                    scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        log.info("S1 Epoch %d done  avg_loss=%.4f  time=%.1fs", epoch, avg_loss, elapsed)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            ckpt_path = output_dir / "stage1_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            log.info("Saved Stage 1 best checkpoint  (loss=%.4f)", best_val_loss)

    model.load_state_dict(torch.load(output_dir / "stage1_best.pt", map_location=device))
    log.info("Stage 1 complete. Best loss: %.4f", best_val_loss)
    return model


@torch.no_grad()
def build_item_embedding_cache(
    model: TwoTowerModel,
    tokenizer,
    anime_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    item_id_to_cf_idx: Optional[dict[int, int]] = None,
) -> tuple[torch.Tensor, dict[int, int]]:
    model.eval()
    e5_prefix = "passage: "

    anime_ids = anime_df["id"].astype(int).tolist()
    texts = [e5_prefix + str(t) for t in anime_df["text_input"].tolist()]
    id_to_idx = {aid: i for i, aid in enumerate(anime_ids)}

    all_embs = []
    bs = cfg.get("s2_encode_batch", 256)
    for i in range(0, len(texts), bs):
        batch_texts = texts[i : i + bs]
        enc = tokenise_batch(batch_texts, tokenizer, cfg["s2_max_length"], device)
        with autocast(enabled=(device.type == "cuda")):
            embs = model.item_tower(**enc)
        all_embs.append(embs.cpu())

    item_matrix = torch.cat(all_embs, dim=0)

    # Fuse CF embeddings if available
    if model.has_cf and item_id_to_cf_idx is not None:
        cf_idxs = torch.tensor(
            [item_id_to_cf_idx.get(aid, 0) for aid in anime_ids],
            dtype=torch.long,
        )
        item_matrix = model.fuse_item_cf(
            item_matrix.to(device), cf_idxs.to(device)
        ).cpu()

    return item_matrix, id_to_idx


def _build_user_batch_context(
    batch: dict,
    item_matrix: torch.Tensor,
    id_to_idx: dict[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_ids = batch["context_ids"]
    context_scores = batch["context_scores"]
    context_mask = batch["context_mask"]

    B, L = context_ids.shape
    D = item_matrix.shape[1]

    flat_ids = context_ids.view(-1).tolist()
    flat_idxs = torch.tensor(
        [id_to_idx.get(aid, 0) for aid in flat_ids],
        dtype=torch.long,
    )
    context_embs = item_matrix[flat_idxs].view(B, L, D).to(device)

    return context_embs, context_scores.to(device), context_mask.to(device)


def train_stage2(
    model: TwoTowerModel,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> tuple[TwoTowerModel, dict]:
    log.info("=" * 60)
    log.info("Stage 2: Two-tower training")
    log.info("=" * 60)

    id_to_text = build_id_to_text(anime_df)

    train_loader = make_user_ratings_loader(
        train_df, id_to_text,
        batch_size=cfg["s2_batch_size"],
        max_history=cfg["s2_max_history"],
        num_workers=cfg.get("num_workers", 2),
        shuffle=True,
    )
    val_loader = make_user_ratings_loader(
        val_df, id_to_text,
        batch_size=cfg["s2_batch_size"],
        max_history=cfg["s2_max_history"],
        num_workers=cfg.get("num_workers", 2),
        shuffle=False,
        min_positives=1,
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))
    model.to(device)

    # ── Freeze ItemTower + item-side CF during Stage 2 ──
    # Stage 1 already fine-tuned the text encoder.  Keeping it frozen means
    # the pre-computed item_matrix is always in sync with the model — no
    # stale-cache problem.  Only UserTower + user CF embeddings are trained.
    for p in model.item_tower.parameters():
        p.requires_grad_(False)
    if model.has_cf:
        model.item_cf_emb.requires_grad_(False)
        model.cf_gate.requires_grad_(False)
    log.info("ItemTower + item CF frozen for Stage 2 — training UserTower + user CF only")

    # Create optimizer AFTER freezing so it only tracks trainable params
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["s2_lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps  = len(train_loader) * cfg["s2_epochs"]
    warmup_steps = cfg.get("s2_warmup_steps", max(1, total_steps // 10))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )

    best_ndcg   = -1.0
    best_metrics = {}
    ks = cfg.get("eval_ks", [5, 10, 20])

    # Build CF ID mappings once (they don't change across epochs)
    item_id_to_cf_idx, user_id_to_cf_idx, _, _ = build_cf_mappings(anime_df, train_df)

    hard_neg_k = cfg.get("s2_hard_neg_k", 0)
    grad_accum = cfg.get("s2_grad_accum", 1)

    # Build item embedding cache ONCE — valid for the entire training run
    # since ItemTower is frozen.
    log.info("  Building item embedding cache (one-time, ItemTower frozen)...")
    item_matrix, id_to_idx = build_item_embedding_cache(
        model, tokenizer, anime_df, cfg, device,
        item_id_to_cf_idx=item_id_to_cf_idx if model.has_cf else None,
    )
    item_matrix_device = item_matrix.to(device)

    for epoch in range(1, cfg["s2_epochs"] + 1):
        model.train()

        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            context_embs, context_scores, context_mask = _build_user_batch_context(
                batch, item_matrix, id_to_idx, device
            )

            # Look up target embeddings from the frozen cache instead of
            # encoding through ItemTower — consistent with context embeddings.
            target_id_list = batch["target_ids"].tolist()
            target_cache_idxs = torch.tensor(
                [id_to_idx.get(int(tid), 0) for tid in target_id_list],
                dtype=torch.long,
            )
            target_embs_cached = item_matrix_device[target_cache_idxs]  # [B, D]

            # Build CF index tensors for the batch
            user_idxs = None
            if model.has_cf:
                user_idxs = torch.tensor(
                    [user_id_to_cf_idx.get(int(uid), 0) for uid in batch["user_ids"].tolist()],
                    dtype=torch.long,
                ).to(device)

            with autocast(enabled=(device.type == "cuda")):
                # Encode user only — target comes from frozen cache
                user_embs = model.encode_user(
                    context_embs, context_scores, context_mask,
                    user_idx=user_idxs,
                )

                # In-batch sampled softmax: user_embs @ target_embs.T
                in_batch_logits = torch.matmul(
                    user_embs, target_embs_cached.T
                ) / model.temperature  # [B, B]

                B = user_embs.size(0)

                if hard_neg_k > 0:
                    # Hard negatives from the frozen item matrix
                    all_scores = torch.matmul(user_embs.detach(), item_matrix_device.T)  # [B, N]

                    # Mask out in-batch targets
                    in_batch_cache_idxs = set(target_cache_idxs.tolist())
                    for idx in in_batch_cache_idxs:
                        all_scores[:, idx] = -1e4

                    # Pick top-K hard negatives per user
                    _, hard_neg_indices = all_scores.topk(hard_neg_k, dim=1)  # [B, K]
                    hard_neg_embs = item_matrix_device[hard_neg_indices]       # [B, K, D]

                    hard_neg_logits = torch.bmm(
                        hard_neg_embs, user_embs.unsqueeze(-1)
                    ).squeeze(-1) / model.temperature  # [B, K]

                    combined_logits = torch.cat([in_batch_logits, hard_neg_logits], dim=1)
                    labels = torch.arange(B, device=combined_logits.device)
                    loss = F.cross_entropy(combined_logits, labels) / grad_accum
                else:
                    labels = torch.arange(B, device=in_batch_logits.device)
                    loss = F.cross_entropy(in_batch_logits, labels) / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("s2_grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum

            if step % 100 == 0:
                log.info(
                    "S2 E%d/%d  step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, cfg["s2_epochs"], step, len(train_loader),
                    loss.item() * grad_accum, scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        log.info("S2 Epoch %d done  avg_loss=%.4f  time=%.1fs", epoch, avg_loss, elapsed)

        # Validation
        metrics = evaluate_epoch(model, tokenizer, val_df, anime_df, cfg, device, ks, train_df=train_df)
        log.info("  Val  %s", format_metrics(metrics))

        ndcg10 = metrics.get("NDCG@10", 0.0)
        if ndcg10 > best_ndcg:
            best_ndcg    = ndcg10
            best_metrics = metrics
            ckpt_path    = output_dir / "stage2_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            log.info("  Saved Stage 2 best checkpoint  NDCG@10=%.4f", best_ndcg)

    model.load_state_dict(torch.load(output_dir / "stage2_best.pt", map_location=device))
    log.info("Stage 2 complete. Best NDCG@10: %.4f", best_ndcg)
    return model, best_metrics


@torch.no_grad()
def evaluate_epoch(
    model: TwoTowerModel,
    tokenizer,
    holdout_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    ks: list[int] = (5, 10, 20),
    train_df: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    model.eval()

    if train_df is None:
        log.warning(
            "evaluate_epoch: train_df not provided — using holdout rows as context. "
            "This underestimates true performance; pass train_df for correct evaluation."
        )
        context_source = holdout_df
    else:
        context_source = train_df

    # Build CF mappings for evaluation
    item_id_to_cf_idx, user_id_to_cf_idx, _, _ = build_cf_mappings(anime_df, context_source)

    item_matrix, id_to_idx = build_item_embedding_cache(
        model, tokenizer, anime_df, cfg, device,
        item_id_to_cf_idx=item_id_to_cf_idx if model.has_cf else None,
    )

    id_to_text = build_id_to_text(anime_df)

    user_to_target: dict[int, int] = {
        int(row["user_id"]): int(row["anime_id"])
        for _, row in holdout_df.iterrows()
    }
    eval_user_ids = list(user_to_target.keys())

    if not eval_user_ids:
        log.warning("evaluate_epoch: holdout_df is empty — returning zero metrics")
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks} | {"MRR": 0.0}

    ctx = context_source[context_source["user_id"].isin(eval_user_ids)].copy()
    target_pairs = holdout_df.set_index("user_id")["anime_id"].to_dict()
    ctx = ctx[~ctx.apply(
        lambda r: int(r["anime_id"]) == int(target_pairs.get(r["user_id"], -1)),
        axis=1,
    )]

    ctx_loader = make_user_ratings_loader(
        ctx, id_to_text,
        batch_size=cfg.get("s2_batch_size", 128),
        max_history=cfg.get("s2_max_history", 50),
        num_workers=0,
        shuffle=False,
        min_positives=1,
    )

    user_emb_map: dict[int, torch.Tensor] = {}

    for batch in ctx_loader:
        ctx_embs, ctx_scores, ctx_mask = _build_user_batch_context(
            batch, item_matrix, id_to_idx, device
        )

        # Build user CF indices for the batch
        user_idx = None
        if model.has_cf:
            user_idx = torch.tensor(
                [user_id_to_cf_idx.get(int(uid), 0) for uid in batch["user_ids"].tolist()],
                dtype=torch.long,
            ).to(device)

        with autocast(enabled=(device.type == "cuda")):
            user_embs = model.encode_user(ctx_embs, ctx_scores, ctx_mask, user_idx=user_idx)

        for uid, emb in zip(batch["user_ids"].tolist(), user_embs.cpu()):
            user_emb_map[uid] = emb

    D = item_matrix.shape[1]
    all_user_embs   = []
    all_target_idxs = []

    for uid in eval_user_ids:
        emb = user_emb_map.get(uid, torch.zeros(D))
        all_user_embs.append(emb)
        target_aid = user_to_target[uid]
        all_target_idxs.append(id_to_idx.get(target_aid, 0))

    if not all_user_embs:
        log.warning("evaluate_epoch: no user embeddings built — returning zero metrics")
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks} | {"MRR": 0.0}

    covered = sum(1 for uid in eval_user_ids if uid in user_emb_map)
    log.info(
        "Evaluation: %d / %d users have context embeddings",
        covered, len(eval_user_ids),
    )

    # Build per-user seen-item index lists for masking
    seen_item_idxs = []
    for uid in eval_user_ids:
        user_ctx = context_source[context_source["user_id"] == uid]
        seen = [
            id_to_idx[int(aid)]
            for aid in user_ctx["anime_id"].tolist()
            if int(aid) in id_to_idx
        ]
        seen_item_idxs.append(seen)

    return evaluate_retrieval(
        user_embeddings=torch.stack(all_user_embs),
        item_embeddings=item_matrix,
        target_item_idxs=torch.tensor(all_target_idxs, dtype=torch.long),
        ks=list(ks),
        seen_item_idxs=seen_item_idxs,
    )


def run_hpo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    base_cfg: dict,
    output_dir: Path,
    device: torch.device,
    n_trials: int = 30,
    encoder_name: str = "intfloat/multilingual-e5-base",
) -> dict:
    """
    Bayesian HPO over Stage 2 hyperparameters using Optuna.

    Search space
    ────────────
    proj_dim      : {64, 128, 256}
    nhead         : {2, 4, 8}  (must divide proj_dim)
    temperature   : [0.03, 0.15]
    s2_lr         : [1e-5, 5e-4]  (log scale)
    s2_batch_size : {128, 256, 512}
    dropout       : [0.0, 0.3]
    weight_decay  : [1e-4, 1e-1] (log scale)
    s2_max_history: {20, 50, 100}
    lora_rank     : {4, 8, 16}   (only when freeze_mode="lora")

    The objective is NDCG@10 on the val split.
    Only Stage 2 is run during HPO (with a reduced epoch count)
    to keep trial time tractable.

    Returns
    -------
    best_params dict (also written to output_dir/hpo_best_params.json)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna:  pip install optuna")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    hpo_epochs = max(1, base_cfg.get("s2_epochs", 5) // 3)

    # Pre-build CF mappings for HPO trials
    item_id_to_cf_idx, user_id_to_cf_idx, n_items_cf, n_users_cf = build_cf_mappings(anime_df, train_df)

    def objective(trial: "optuna.Trial") -> float:
        proj_dim = trial.suggest_categorical("proj_dim", [64, 128, 256])

        valid_nheads = [n for n in [2, 4, 8] if proj_dim % n == 0]
        nhead = trial.suggest_categorical("nhead", valid_nheads)

        cf_dim = trial.suggest_categorical("cf_dim", [0, 32, 64, 128])
        user_tower_layers = trial.suggest_int("user_tower_layers", 1, 3)
        s2_hard_neg_k = trial.suggest_categorical("s2_hard_neg_k", [0, 64, 128, 256])

        cfg = {
            **base_cfg,
            "proj_dim": proj_dim,
            "nhead": nhead,
            "cf_dim": cf_dim,
            "user_tower_layers": user_tower_layers,
            "s2_hard_neg_k": s2_hard_neg_k,
            "temperature": trial.suggest_float("temperature",  0.03, 0.15),
            "s2_lr": trial.suggest_float("s2_lr",        1e-5, 5e-4, log=True),
            "s2_batch_size": trial.suggest_categorical("s2_batch_size", [128, 256, 512]),
            "dropout": trial.suggest_float("dropout",       0.0,  0.3),
            "weight_decay": trial.suggest_float("weight_decay",  1e-4, 1e-1, log=True),
            "s2_max_history": trial.suggest_categorical("s2_max_history", [20, 50, 100]),
            "lora_rank": trial.suggest_categorical("lora_rank", [4, 8, 16]),
            "s2_epochs": hpo_epochs,
            "s2_warmup_steps": 50,
        }

        trial_dir = output_dir / f"hpo_trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = TwoTowerModel(
            encoder_name=encoder_name,
            proj_dim=cfg["proj_dim"],
            nhead=cfg["nhead"],
            temperature=cfg["temperature"],
            dropout=cfg["dropout"],
            freeze_mode=base_cfg.get("freeze_mode", "lora"),
            lora_rank=cfg["lora_rank"],
            lora_alpha=float(cfg["lora_rank"]) * 2,
            n_items=n_items_cf if cf_dim > 0 else 0,
            n_users=n_users_cf if cf_dim > 0 else 0,
            cf_dim=cf_dim,
            user_tower_layers=user_tower_layers,
        )

        try:
            _, metrics = train_stage2(
                model, tokenizer, train_df, val_df, anime_df,
                cfg, trial_dir, device,
            )
            ndcg10 = metrics.get("NDCG@10", 0.0)
        except Exception as e:
            log.warning("Trial %d failed: %s", trial.number, e)
            return 0.0

        log.info("Trial %d  NDCG@10=%.4f  params=%s",
                 trial.number, ndcg10, trial.params)
        return ndcg10

    log.info("Starting HPO  (n_trials=%d, hpo_epochs_per_trial=%d)", n_trials, hpo_epochs)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value  = study.best_value
    log.info("HPO complete. Best NDCG@10=%.4f  params=%s", best_value, best_params)

    out_path = output_dir / "hpo_best_params.json"
    with open(out_path, "w") as f:
        json.dump({"best_value": best_value, "best_params": best_params}, f, indent=2)
    log.info("Saved HPO results to %s", out_path)

    try:
        importance = optuna.importance.get_param_importances(study)
        log.info("Hyperparameter importances:")
        for name, score in sorted(importance.items(), key=lambda x: -x[1]):
            log.info("  %-20s  %.3f", name, score)
    except Exception:
        pass

    return best_params


def run_hpo_reranker(
    two_tower_model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    base_cfg: dict,
    output_dir: Path,
    device: torch.device,
    n_trials: int = 20,
    reranker_encoder: Optional[str] = None,
) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    from model.reranker import (
        CrossEncoderReranker, TwoTowerWithReranker,
        train_stage3,
    )
    from model.metrics import evaluate_reranker

    if reranker_encoder is None:
        reranker_encoder = base_cfg.get("s3_encoder", "BAAI/bge-reranker-v2-m3")
    s3_tokenizer = AutoTokenizer.from_pretrained(reranker_encoder)
    hpo_epochs = max(1, base_cfg.get("s3_epochs", 3) // 2)

    # Build CF mappings for reranker HPO
    _item_cf, _user_cf, _, _ = build_cf_mappings(anime_df, train_df)

    log.info("Pre-building two-tower catalogue for reranker HPO...")
    _tmp_recommender = TwoTowerWithReranker(
        two_tower=two_tower_model,
        reranker=CrossEncoderReranker(
            encoder_name=reranker_encoder,
            pretrained_reranker=base_cfg.get("s3_pretrained_reranker", False),
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            base_cfg.get("encoder", "intfloat/multilingual-e5-base")
        ),
        reranker_tokenizer=s3_tokenizer,
        anime_df=anime_df,
        device=device,
        item_id_to_cf_idx=_item_cf if two_tower_model.has_cf else None,
        user_id_to_cf_idx=_user_cf if two_tower_model.has_cf else None,
    )
    item_matrix = _tmp_recommender.item_matrix
    id_to_idx = _tmp_recommender.id_to_idx
    idx_to_id = _tmp_recommender.idx_to_id
    id_to_text = _tmp_recommender.id_to_text
    id_to_name = _tmp_recommender.id_to_name
    del _tmp_recommender

    def objective(trial: "optuna.Trial") -> float:
        cfg = {
            **base_cfg,
            "s3_lr": trial.suggest_float("s3_lr",          1e-6, 1e-4, log=True),
            # "s3_batch_size": trial.suggest_categorical("s3_batch_size",   [8, 16, 32]),
            "s3_grad_accum": trial.suggest_categorical("s3_grad_accum",   [2, 4, 8]),
            "s3_neg_per_user": trial.suggest_categorical("s3_neg_per_user", [1, 3, 5]),
            "s3_max_length": trial.suggest_categorical("s3_max_length",   [128, 256]),
            "dropout": trial.suggest_float("dropout",          0.0,  0.3),
            # "retrieval_k": trial.suggest_categorical("retrieval_k",     [50, 100, 200]),
            "s3_epochs": hpo_epochs,
            "s3_warmup_steps": 20,
        }

        trial_dir = output_dir / f"hpo_s3_trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        reranker = CrossEncoderReranker(
            encoder_name=reranker_encoder,
            dropout=cfg["dropout"],
            pretrained_reranker=cfg.get("s3_pretrained_reranker", False),
        )

        try:
            reranker = train_stage3(
                reranker, s3_tokenizer,
                train_df=train_df, val_df=val_df,
                anime_df=anime_df, cfg=cfg,
                output_dir=trial_dir, device=device,
            )
        except Exception as e:
            log.warning("S3 HPO trial %d training failed: %s", trial.number, e)
            return 0.0

        recommender = TwoTowerWithReranker.__new__(TwoTowerWithReranker)
        recommender.two_tower = two_tower_model
        recommender.reranker = reranker
        recommender.tokenizer = AutoTokenizer.from_pretrained(
            base_cfg.get("encoder", "intfloat/multilingual-e5-base")
        )
        recommender.reranker_tokenizer = s3_tokenizer
        recommender.device = device
        recommender.max_length = base_cfg.get("max_length", 128)
        recommender.rerank_max_length = cfg["s3_max_length"]
        recommender.anime_df = anime_df
        recommender.item_matrix = item_matrix
        recommender.id_to_idx = id_to_idx
        recommender.idx_to_id = idx_to_id
        recommender.id_to_text = id_to_text
        recommender.id_to_name = id_to_name
        recommender.item_id_to_cf_idx = _item_cf if two_tower_model.has_cf else {}
        recommender.user_id_to_cf_idx = _user_cf if two_tower_model.has_cf else {}

        try:
            metrics = evaluate_reranker(
                recommender=recommender,
                holdout_df=val_df,
                train_df=train_df,
                ks=[10],
                retrieval_k=cfg["retrieval_k"],
                batch_size=64,
            )
            ndcg10 = metrics.get("NDCG@10", 0.0)
        except Exception as e:
            log.warning("S3 HPO trial %d eval failed: %s", trial.number, e)
            return 0.0

        log.info("S3 Trial %d  NDCG@10=%.4f  params=%s",
                 trial.number, ndcg10, trial.params)

        # Clean up trial reranker to free VRAM
        del reranker, recommender
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return ndcg10

    log.info("Starting reranker HPO  (n_trials=%d, epochs_per_trial=%d)",
             n_trials, hpo_epochs)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value  = study.best_value
    log.info("Reranker HPO complete. Best NDCG@10=%.4f  params=%s",
             best_value, best_params)

    out_path = output_dir / "hpo_reranker_best_params.json"
    with open(out_path, "w") as f:
        json.dump({"best_value": best_value, "best_params": best_params}, f, indent=2)
    log.info("Saved reranker HPO results to %s", out_path)

    try:
        importance = optuna.importance.get_param_importances(study)
        log.info("Reranker HPO param importances:")
        for name, score in sorted(importance.items(), key=lambda x: -x[1]):
            log.info("  %-20s  %.3f", name, score)
    except Exception:
        pass

    return best_params

DEFAULT_CFG = {
    # Stage 1
    "s1_epochs": 7,
    "s1_batch_size": 32,
    "s1_grad_accum": 4, 
    "s1_lr": 2e-5,
    "s1_warmup_steps": 100,
    "s1_max_length": 512,
    "s1_neg_per_user": 5,
    "s1_grad_clip": 1.0,
    # Stage 2
    "s2_epochs": 10,
    "s2_batch_size": 64,
    "s2_grad_accum": 2,
    "s2_lr": 3e-5,
    "s2_warmup_steps": 200,
    "s2_max_length": 512,
    "s2_max_history": 50,
    "s2_grad_clip": 1.0,
    "s2_encode_batch": 128,
    "s2_cache_refresh_steps": 50,
    # Shared
    "proj_dim": 256,
    "nhead": 4,
    "temperature": 0.07,
    "dropout": 0.1,
    "weight_decay": 0.01,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.05,
    "freeze_mode": "lora",
    "encoder": "intfloat/multilingual-e5-base",
    "pooling": "mean",
    "cf_dim": 128,
    "user_tower_layers": 2,
    "s2_hard_neg_k": 128,
    "eval_ks": [5, 10, 20],
    "num_workers": 2,
    # Stage 3 — cross-encoder reranker
    "s3_encoder": "BAAI/bge-reranker-v2-m3",
    "s3_pretrained_reranker": True,
    "s3_epochs": 3,
    "s3_batch_size": 32,
    "s3_grad_accum": 2,
    "s3_lr": 2e-5,
    "s3_warmup_steps": 50,
    "s3_max_length": 512,
    "s3_grad_clip": 1.0,
    "s3_neg_per_user": 3,
    "retrieval_k": 100,
}


def main():
    parser = argparse.ArgumentParser(description="Train the two-tower anime recommender")

    parser.add_argument("--processed_dir", default="./processed_data",
                        help="Output of preprocessing.py")
    parser.add_argument("--output_dir",    default="./checkpoints",
                        help="Where to save model checkpoints and HPO results")
    parser.add_argument("--encoder",       default="intfloat/multilingual-e5-base",
                        help="HuggingFace encoder model ID")
    parser.add_argument("--freeze_mode",   default="lora",
                        choices=["all", "lora", "none"],
                        help="Encoder freeze strategy")

    # Stage toggles
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip contrastive fine-tuning (Stage 1)")
    parser.add_argument("--skip_stage2", action="store_true",
                        help="Skip two-tower training (Stage 2)")
    parser.add_argument("--skip_stage3", action="store_true",
                        help="Skip cross-encoder reranker training (Stage 3)")

    parser.add_argument("--hpo", action="store_true",
                        help="Run Optuna HPO before the final training run")
    parser.add_argument("--hpo_trials", type=int, default=30,
                        help="Number of Optuna trials")

    # Quick-override of key hypers (optional)
    parser.add_argument("--proj_dim", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--s2_lr", type=float, default=None)
    parser.add_argument("--s2_batch_size", type=int, default=None)
    parser.add_argument("--s2_epochs", type=int, default=None)
    parser.add_argument("--s1_epochs", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading processed data from %s", processed_dir)
    anime_df = pd.read_parquet(processed_dir / "anime_processed.parquet")
    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")
    test_df  = pd.read_parquet(processed_dir / "test_interactions.parquet")

    log.info(
        "Loaded — anime: %d  train: %d  val: %d  test: %d",
        len(anime_df), len(train_df), len(val_df), len(test_df),
    )

    cfg = {**DEFAULT_CFG}
    cfg["freeze_mode"] = args.freeze_mode
    for key in ["proj_dim", "nhead", "temperature", "s2_lr", "s2_batch_size",
                "s2_epochs", "s1_epochs", "lora_rank"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    if args.hpo:
        best_params = run_hpo(
            train_df, val_df, anime_df,
            base_cfg=cfg,
            output_dir=output_dir,
            device=device,
            n_trials=args.hpo_trials,
            encoder_name=args.encoder,
        )
        cfg.update(best_params)
        cfg["s2_epochs"] = DEFAULT_CFG["s2_epochs"]
        cfg["s1_epochs"] = DEFAULT_CFG["s1_epochs"]
        log.info("Running final training with HPO-optimised hyperparameters")

    # Save final config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Tokeniser ──
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # ── Stage 1: Contrastive fine-tuning ──
    stage1_weights_path = output_dir / "stage1_best.pt"

    if not args.skip_stage1:
        s1_model = ContrastiveEncoder(
            encoder_name=args.encoder,
            proj_dim=cfg["proj_dim"],
            dropout=cfg["dropout"],
            freeze_mode=cfg["freeze_mode"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg.get("lora_alpha", 16.0),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            base_margin=cfg.get("base_margin", 0.3),
        )
        s1_model = train_stage1(
            s1_model, tokenizer, train_df, val_df, anime_df,
            cfg, output_dir, device,
        )
        torch.save(s1_model.tower.state_dict(), stage1_weights_path)
        log.info("Stage 1 encoder weights saved to %s", stage1_weights_path)
    else:
        log.info("Skipping Stage 1 (--skip_stage1)")

    if not args.skip_stage2:
        # Build CF mappings
        item_id_to_cf_idx, user_id_to_cf_idx, n_items_cf, n_users_cf = build_cf_mappings(anime_df, train_df)
        cf_dim = cfg.get("cf_dim", 0)
        if cf_dim == 0:
            n_items_cf = 0
            n_users_cf = 0

        model = TwoTowerModel(
            encoder_name=args.encoder,
            proj_dim=cfg["proj_dim"],
            nhead=cfg["nhead"],
            temperature=cfg["temperature"],
            dropout=cfg["dropout"],
            freeze_mode=cfg["freeze_mode"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg.get("lora_alpha", 16.0),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            pooling=cfg.get("pooling", "mean"),
            n_items=n_items_cf,
            n_users=n_users_cf,
            cf_dim=cf_dim,
            user_tower_layers=cfg.get("user_tower_layers", 1),
        )

        if stage1_weights_path.exists() and not args.skip_stage1:
            log.info("Loading Stage 1 encoder weights into item tower...")
            state = torch.load(stage1_weights_path, map_location="cpu")
            missing, unexpected = model.item_tower.load_state_dict(state, strict=False)
            if missing:
                log.warning("Missing keys when loading S1→S2: %s", missing)
            if unexpected:
                log.warning("Unexpected keys when loading S1→S2: %s", unexpected)

        model, best_val_metrics = train_stage2(
            model, tokenizer, train_df, val_df, anime_df,
            cfg, output_dir, device,
        )

        log.info("Running final evaluation on held-out test set...")
        train_and_val = pd.concat([train_df, val_df], ignore_index=True)
        test_metrics = evaluate_epoch(
            model, tokenizer, test_df, anime_df, cfg, device,
            ks=cfg["eval_ks"],
            train_df=train_and_val,
        )
        log.info("TEST METRICS: %s", format_metrics(test_metrics, prefix=""))

        results = {
            "val_metrics":  best_val_metrics,
            "test_metrics": test_metrics,
            "config":       cfg,
        }
        with open(output_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info("Final results saved to %s/final_results.json", output_dir)

        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), final_model_dir / "model.pt")
        tokenizer.save_pretrained(str(final_model_dir))
        with open(final_model_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        log.info("Inference-ready model saved to %s", final_model_dir)

    if not args.skip_stage3 and not args.skip_stage2:
        log.info("Training Stage 3: cross-encoder reranker...")
        s3_tokenizer = AutoTokenizer.from_pretrained(cfg["s3_encoder"])
        reranker = CrossEncoderReranker(
            encoder_name=cfg["s3_encoder"],
            dropout=cfg.get("dropout", 0.1),
            pretrained_reranker=cfg.get("s3_pretrained_reranker", False),
        )
        reranker = train_stage3(
            reranker, s3_tokenizer,
            train_df=train_df, val_df=val_df,
            anime_df=anime_df, cfg=cfg,
            output_dir=output_dir, device=device,
        )
        reranker_dir = output_dir / "final_model"
        reranker_dir.mkdir(exist_ok=True)
        torch.save(reranker.state_dict(), reranker_dir / "reranker.pt")
        s3_tokenizer.save_pretrained(str(reranker_dir / "reranker_tokenizer"))
        log.info("Reranker saved to %s", reranker_dir)
    elif args.skip_stage3:
        log.info("Skipping Stage 3 (--skip_stage3)")

    log.info("All done.")


if __name__ == "__main__":
    main()