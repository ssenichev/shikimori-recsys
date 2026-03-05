"""
Full training pipeline with Hydra configuration.

Usage:
    # Default config
    python main.py

    # Override params from CLI
    python main.py encoder=intfloat/multilingual-e5-large stage2.batch_size=128

    # Skip stages
    python main.py skip_stage1=true skip_stage3=true

    # Run HPO
    python main.py hpo.tower=true hpo.tower_trials=30

    # Use a different config file
    python main.py --config-name=my_experiment
"""

import json
import logging
import os
import pathlib

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data.preprocessing import run_preprocessing
from model.architecture import ContrastiveEncoder, TwoTowerModel
from model.reranker import CrossEncoderReranker, TwoTowerWithReranker, train_stage3
from model.train import (
    train_stage1,
    train_stage2,
    evaluate_epoch,
    build_cf_mappings,
    run_hpo,
    run_hpo_reranker,
    DEFAULT_CFG,
)
from model.metrics import evaluate_reranker, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_train_cfg(hydra_cfg: DictConfig) -> dict:
    """Flatten Hydra config into the flat dict expected by train functions."""
    cfg = {**DEFAULT_CFG}

    cfg["encoder"] = hydra_cfg.encoder
    cfg["freeze_mode"] = hydra_cfg.freeze_mode
    cfg["pooling"] = hydra_cfg.pooling

    # Stage 1
    s1 = hydra_cfg.stage1
    cfg["s1_epochs"] = s1.epochs
    cfg["s1_batch_size"] = s1.batch_size
    cfg["s1_grad_accum"] = s1.grad_accum
    cfg["s1_lr"] = s1.lr
    cfg["s1_warmup_steps"] = s1.warmup_steps
    cfg["s1_max_length"] = s1.max_length
    cfg["s1_neg_per_user"] = s1.neg_per_user
    cfg["s1_grad_clip"] = s1.grad_clip

    # Stage 2
    s2 = hydra_cfg.stage2
    cfg["s2_epochs"] = s2.epochs
    cfg["s2_batch_size"] = s2.batch_size
    cfg["s2_grad_accum"] = s2.grad_accum
    cfg["s2_lr"] = s2.lr
    cfg["s2_warmup_steps"] = s2.warmup_steps
    cfg["s2_max_length"] = s2.max_length
    cfg["s2_max_history"] = s2.max_history
    cfg["s2_grad_clip"] = s2.grad_clip
    cfg["s2_encode_batch"] = s2.encode_batch
    cfg["s2_hard_neg_k"] = s2.hard_neg_k

    # Model
    m = hydra_cfg.model
    cfg["proj_dim"] = m.proj_dim
    cfg["nhead"] = m.nhead
    cfg["temperature"] = m.temperature
    cfg["dropout"] = m.dropout
    cfg["weight_decay"] = m.weight_decay
    cfg["lora_rank"] = m.lora_rank
    cfg["lora_alpha"] = m.lora_alpha
    cfg["lora_dropout"] = m.lora_dropout
    cfg["lora_targets"] = m.lora_targets
    cfg["cf_dim"] = m.cf_dim
    cfg["user_tower_layers"] = m.user_tower_layers

    # Stage 3
    s3 = hydra_cfg.stage3
    cfg["s3_encoder"] = s3.encoder
    cfg["s3_pretrained_reranker"] = s3.pretrained_reranker
    cfg["s3_epochs"] = s3.epochs
    cfg["s3_batch_size"] = s3.batch_size
    cfg["s3_grad_accum"] = s3.grad_accum
    cfg["s3_lr"] = s3.lr
    cfg["s3_warmup_steps"] = s3.warmup_steps
    cfg["s3_max_length"] = s3.max_length
    cfg["s3_grad_clip"] = s3.grad_clip
    cfg["s3_neg_per_user"] = s3.neg_per_user
    cfg["retrieval_k"] = s3.retrieval_k

    # Eval
    cfg["eval_ks"] = list(hydra_cfg.eval.ks)
    cfg["num_workers"] = hydra_cfg.num_workers

    return cfg


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(hydra_cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(hydra_cfg))

    # Hydra changes cwd to outputs/<date>/<time>, resolve paths relative to original cwd
    original_cwd = hydra.utils.get_original_cwd()

    # Reproducibility
    torch.manual_seed(hydra_cfg.seed)
    np.random.seed(hydra_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    raw_dir = os.path.join(original_cwd, hydra_cfg.data.raw_dir)
    processed_dir = os.path.join(original_cwd, hydra_cfg.data.processed_dir)
    output_dir = pathlib.Path(os.path.join(original_cwd, hydra_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    if hydra_cfg.data.run_preprocessing:
        log.info("Running preprocessing...")
        stats = run_preprocessing(data_dir=raw_dir, output_dir=processed_dir)
        log.info("Stats:\n%s", json.dumps(stats, indent=2))
    else:
        log.info("Skipping preprocessing (data.run_preprocessing=false)")

    log.info("Loading processed data from %s", processed_dir)
    anime_df = pd.read_parquet(os.path.join(processed_dir, "anime_processed.parquet"))
    train_df = pd.read_parquet(os.path.join(processed_dir, "train_interactions.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val_interactions.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test_interactions.parquet"))
    log.info(
        "Loaded — anime: %d  train: %d  val: %d  test: %d",
        len(anime_df), len(train_df), len(val_df), len(test_df),
    )

    # ── Build flat config ──
    cfg = _build_train_cfg(hydra_cfg)
    tokenizer = AutoTokenizer.from_pretrained(hydra_cfg.encoder)

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── HPO (Stage 2) ──
    if hydra_cfg.hpo.tower:
        log.info("Running Stage 2 HPO (%d trials)...", hydra_cfg.hpo.tower_trials)
        best_params = run_hpo(
            train_df=train_df,
            val_df=val_df,
            anime_df=anime_df,
            base_cfg=cfg,
            output_dir=output_dir,
            device=device,
            n_trials=hydra_cfg.hpo.tower_trials,
            encoder_name=hydra_cfg.encoder,
        )
        cfg.update(best_params)
        # Restore epoch counts (HPO uses reduced epochs)
        cfg["s1_epochs"] = hydra_cfg.stage1.epochs
        cfg["s2_epochs"] = hydra_cfg.stage2.epochs
        log.info("HPO best params: %s", best_params)

    # ── Stage 1: Contrastive fine-tuning ──
    stage1_path = output_dir / "stage1_encoder.pt"

    if not hydra_cfg.skip_stage1:
        log.info("=" * 60)
        log.info("STAGE 1: Contrastive fine-tuning")
        log.info("=" * 60)

        s1_model = ContrastiveEncoder(
            encoder_name=hydra_cfg.encoder,
            proj_dim=cfg["proj_dim"],
            dropout=cfg["dropout"],
            freeze_mode=cfg["freeze_mode"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            lora_targets=cfg["lora_targets"],
            base_margin=cfg.get("base_margin", 0.3),
        )
        s1_model = train_stage1(
            s1_model, tokenizer,
            train_df=train_df, val_df=val_df, anime_df=anime_df,
            cfg=cfg, output_dir=output_dir, device=device,
        )
        torch.save(s1_model.tower.state_dict(), stage1_path)
        log.info("Stage 1 encoder weights saved to %s", stage1_path)
        del s1_model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        log.info("Skipping Stage 1 (skip_stage1=true)")

    # ── Stage 2: Two-Tower training ──
    if not hydra_cfg.skip_stage2:
        log.info("=" * 60)
        log.info("STAGE 2: Two-Tower training")
        log.info("=" * 60)

        item_id_to_cf_idx, user_id_to_cf_idx, n_items_cf, n_users_cf = build_cf_mappings(
            anime_df, train_df,
        )
        cf_dim = cfg["cf_dim"]
        if cf_dim == 0:
            n_items_cf = 0
            n_users_cf = 0

        model = TwoTowerModel(
            encoder_name=hydra_cfg.encoder,
            proj_dim=cfg["proj_dim"],
            nhead=cfg["nhead"],
            temperature=cfg["temperature"],
            dropout=cfg["dropout"],
            freeze_mode=cfg["freeze_mode"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            lora_targets=cfg["lora_targets"],
            pooling=cfg["pooling"],
            n_items=n_items_cf,
            n_users=n_users_cf,
            cf_dim=cf_dim,
            user_tower_layers=cfg["user_tower_layers"],
        )

        # Load Stage 1 weights into ItemTower
        if stage1_path.exists():
            log.info("Loading Stage 1 encoder weights into ItemTower...")
            state = torch.load(stage1_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.item_tower.load_state_dict(state, strict=False)
            if missing:
                log.warning("Missing keys: %s", missing)
            if unexpected:
                log.warning("Unexpected keys: %s", unexpected)
        else:
            log.warning("No Stage 1 checkpoint found at %s — training from scratch", stage1_path)

        model, best_val_metrics = train_stage2(
            model, tokenizer,
            train_df=train_df, val_df=val_df, anime_df=anime_df,
            cfg=cfg, output_dir=output_dir, device=device,
        )
        log.info("Best val metrics: %s", format_metrics(best_val_metrics))

        # ── Test evaluation ──
        log.info("Evaluating on test set...")
        train_and_val = pd.concat([train_df, val_df], ignore_index=True)
        test_metrics = evaluate_epoch(
            model, tokenizer, test_df, anime_df,
            cfg=cfg, device=device,
            ks=cfg["eval_ks"],
            train_df=train_and_val,
        )
        log.info("TEST METRICS: %s", format_metrics(test_metrics))

        # ── Save model ──
        final_dir = output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), final_dir / "model.pt")
        tokenizer.save_pretrained(str(final_dir))
        with open(final_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        results = {
            "val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "config": cfg,
        }
        with open(output_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info("Model saved to %s", final_dir)
    else:
        log.info("Skipping Stage 2 (skip_stage2=true)")
        model = None

    # ── Stage 3: Cross-encoder reranker ──
    if not hydra_cfg.skip_stage3 and model is not None:
        log.info("=" * 60)
        log.info("STAGE 3: Cross-encoder reranker")
        log.info("=" * 60)

        # Reranker HPO
        if hydra_cfg.hpo.reranker:
            log.info("Running reranker HPO (%d trials)...", hydra_cfg.hpo.reranker_trials)
            best_s3_params = run_hpo_reranker(
                two_tower_model=model,
                train_df=train_df,
                val_df=val_df,
                anime_df=anime_df,
                base_cfg=cfg,
                output_dir=output_dir,
                device=device,
                n_trials=hydra_cfg.hpo.reranker_trials,
            )
            cfg.update(best_s3_params)
            log.info("Reranker HPO best params: %s", best_s3_params)

        s3_tokenizer = AutoTokenizer.from_pretrained(cfg["s3_encoder"])
        reranker = CrossEncoderReranker(
            encoder_name=cfg["s3_encoder"],
            dropout=cfg["dropout"],
            pretrained_reranker=cfg["s3_pretrained_reranker"],
        )
        reranker = train_stage3(
            reranker, s3_tokenizer,
            train_df=train_df, val_df=val_df, anime_df=anime_df,
            cfg=cfg, output_dir=output_dir, device=device,
        )

        # Save reranker
        final_dir = output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        torch.save(reranker.state_dict(), final_dir / "reranker.pt")
        s3_tokenizer.save_pretrained(str(final_dir / "reranker_tokenizer"))
        log.info("Reranker saved to %s", final_dir)

        # ── Evaluate reranker ──
        log.info("Building recommender for evaluation...")
        recommender = TwoTowerWithReranker(
            two_tower=model,
            reranker=reranker,
            tokenizer=tokenizer,
            reranker_tokenizer=s3_tokenizer,
            anime_df=anime_df,
            device=device,
            item_id_to_cf_idx=item_id_to_cf_idx if model.has_cf else None,
            user_id_to_cf_idx=user_id_to_cf_idx if model.has_cf else None,
        )

        log.info("Evaluating reranker on val set...")
        val_reranker_metrics = evaluate_reranker(
            recommender=recommender,
            holdout_df=val_df,
            train_df=train_df,
            ks=cfg["eval_ks"],
            retrieval_k=cfg["retrieval_k"],
        )
        log.info("RERANKER VAL: %s", format_metrics(val_reranker_metrics))

        log.info("Evaluating reranker on test set...")
        train_and_val = pd.concat([train_df, val_df], ignore_index=True)
        test_reranker_metrics = evaluate_reranker(
            recommender=recommender,
            holdout_df=test_df,
            train_df=train_and_val,
            ks=cfg["eval_ks"],
            retrieval_k=cfg["retrieval_k"],
        )
        log.info("RERANKER TEST: %s", format_metrics(test_reranker_metrics))

        # Save all results
        all_results = {
            "val_metrics": best_val_metrics if not hydra_cfg.skip_stage2 else {},
            "test_metrics": test_metrics if not hydra_cfg.skip_stage2 else {},
            "val_reranker_metrics": val_reranker_metrics,
            "test_reranker_metrics": test_reranker_metrics,
            "config": cfg,
        }
        with open(output_dir / "final_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        if hydra_cfg.skip_stage3:
            log.info("Skipping Stage 3 (skip_stage3=true)")
        elif model is None:
            log.info("Skipping Stage 3 (no two-tower model available)")

    log.info("All done.")


if __name__ == "__main__":
    main()
