import numpy as np
import torch
from typing import Optional


def hit_rate_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    return float(np.mean(dcg))


def mrr(ranks: np.ndarray) -> float:
    valid = ranks[np.isfinite(ranks) & (ranks > 0)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(1.0 / valid))


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return hit_rate_at_k(ranks, k)


def evaluate_retrieval(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    target_item_idxs: torch.Tensor,
    ks: list[int] = (5, 10, 20),
    batch_size: int = 512,
) -> dict[str, float]:
    user_embeddings = user_embeddings.float()
    item_embeddings = item_embeddings.float()
    target_item_idxs = target_item_idxs.long()

    user_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=-1)
    item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)

    ranks = []
    n_users = user_embeddings.size(0)

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        u_batch = user_embeddings[start:end]
        targets = target_item_idxs[start:end]

        scores = torch.matmul(u_batch, item_embeddings.T)
        sorted_idxs = torch.argsort(scores, dim=1, descending=True)

        for i in range(end - start):
            target = targets[i].item()
            pos = (sorted_idxs[i] == target).nonzero(as_tuple=True)[0]
            rank = int(pos[0].item()) + 1 if len(pos) > 0 else float("inf")
            ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=float)
    results = {"MRR": mrr(ranks_arr)}
    
    for k in ks:
        results[f"HR@{k}"]   = hit_rate_at_k(ranks_arr, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranks_arr, k)

    return results


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    parts = [f"{prefix}{k}: {v:.4f}" for k, v in sorted(metrics.items())]
    return "  ".join(parts)


def evaluate_reranker(
    recommender: "TwoTowerWithReranker", # noqa: F821
    holdout_df: "pd.DataFrame", # noqa: F821
    train_df: "pd.DataFrame", # noqa: F821
    ks: list[int] = (5, 10, 20),
    retrieval_k: int = 100,
    batch_size: int = 32,
) -> dict[str, float]:
    
    import pandas as pd
    import torch
    import logging
    log = logging.getLogger(__name__)

    device = recommender.device
    item_matrix = recommender.item_matrix
    id_to_idx = recommender.id_to_idx

    user_to_target = {
        int(row["user_id"]): int(row["anime_id"])
        for _, row in holdout_df.iterrows()
    }
    eval_user_ids = list(user_to_target.keys())
    log.info("Evaluating reranker on %d users (retrieval_k=%d)", len(eval_user_ids), retrieval_k)

    ranks_reranked = []
    in_candidates = 0

    for i in range(0, len(eval_user_ids), batch_size):
        batch_user_ids = eval_user_ids[i : i + batch_size]

        for uid in batch_user_ids:
            target_aid = user_to_target[uid]

            ctx = train_df[
                (train_df["user_id"] == uid) &
                (train_df["anime_id"] != target_aid)
            ]
            if ctx.empty:
                ranks_reranked.append(float("inf"))
                continue

            try:
                user_vec = recommender._encode_user(ctx)
            except Exception as e:
                log.warning("User %d encode failed: %s", uid, e)
                ranks_reranked.append(float("inf"))
                continue

            scores = (user_vec.unsqueeze(0) @ item_matrix.T).squeeze(0)

            seen_ids = set(ctx["anime_id"].astype(int).tolist())
            for sid in seen_ids:
                if sid in id_to_idx:
                    scores[id_to_idx[sid]] = -1e9

            top_idxs = scores.topk(retrieval_k).indices.tolist()
            candidate_ids = [recommender.idx_to_id[idx] for idx in top_idxs]

            if target_aid in candidate_ids:
                in_candidates += 1
            else:
                # Reranker cannot help if retriever missed the target
                ranks_reranked.append(float("inf"))
                continue

            profile_text = recommender.id_to_name and __import__("model.reranker", fromlist=["build_user_profile_text"]).build_user_profile_text(ctx, recommender.id_to_name)

            try:
                reranked = recommender._rerank(profile_text, candidate_ids)
            except Exception as e:
                log.warning("User %d rerank failed: %s", uid, e)
                ranks_reranked.append(float("inf"))
                continue

            reranked_ids = [aid for aid, _ in reranked]
            if target_aid in reranked_ids:
                rank = reranked_ids.index(target_aid) + 1
            else:
                rank = float("inf")

            ranks_reranked.append(rank)

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(eval_user_ids))
            log.info("  %d / %d users evaluated...", done, len(eval_user_ids))

    ranks_arr = np.array(ranks_reranked, dtype=float)

    results = {
        "MRR":              mrr(ranks_arr),
        "retrieval_recall": in_candidates / max(len(eval_user_ids), 1),
    }
    for k in ks:
        results[f"HR@{k}"]   = hit_rate_at_k(ranks_arr, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranks_arr, k)

    log.info(
        in_candidates, len(eval_user_ids),
        100 * in_candidates / max(len(eval_user_ids), 1),
    )
    return results
