import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SCORE_HIGH = 7
SCORE_LOW  = 4


def build_id_to_text(anime_df: pd.DataFrame) -> dict[int, str]:
    return {
        int(row["id"]): str(row["text_input"])
        for _, row in anime_df.iterrows()
    }


class AnimeTextDataset(Dataset):
    def __init__(self, anime_df: pd.DataFrame, e5_prefix: str = "passage: "):
        self.records = [
            (int(row["id"]), e5_prefix + str(row["text_input"]))
            for _, row in anime_df.iterrows()
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        anime_id, text = self.records[idx]
        return {"anime_id": anime_id, "text": text}


class UserRatingsDataset(Dataset):
    """
    interactions : DataFrame with columns
                   [user_id, anime_id, score_raw, score_norm, confidence]
    id_to_text : mapping from anime_id to text string (for on-the-fly text lookup)
    max_history : maximum number of context items per user
    min_positives: users with fewer than this many high-scored items are skipped
    e5_prefix : prefix applied to candidate text for multilingual-e5
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        id_to_text: dict[int, str],
        max_history: int = 50,
        min_positives: int = 2,
        e5_prefix: str = "passage: ",
    ):
        self.id_to_text = id_to_text
        self.max_history = max_history
        self.e5_prefix = e5_prefix

        # Group by user, keep only users with enough explicit positives.
        # Implicit interactions (is_explicit=False) are included in the
        # context window but never used as retrieval targets.
        self.samples: list[dict] = []
        for user_id, group in interactions.groupby("user_id"):
            # Sort context by created_at (chronological) so attention over
            # history sees items in the order they were watched.
            if "created_at" in group.columns:
                group = group.sort_values("created_at", ascending=True)
            else:
                group = group.sort_values("confidence", ascending=False)

            # Targets must be explicit ratings
            if "is_explicit" in group.columns:
                positives = group[group["is_explicit"] & (group["score_raw"] >= SCORE_HIGH)]
            else:
                positives = group[group["score_raw"] >= SCORE_HIGH]

            if len(positives) < min_positives:
                continue

            # For each explicit positive build one training sample.
            # Context = all other items (explicit + implicit) the user interacted
            # with before the target, up to max_history.
            for _, pos_row in positives.iterrows():
                target_id = int(pos_row["anime_id"])

                # Use chronological order: only items BEFORE the target
                if "created_at" in group.columns:
                    context = group[
                        (group["anime_id"] != target_id) &
                        (group["created_at"] <= pos_row["created_at"])
                    ].tail(max_history)
                else:
                    context = group[group["anime_id"] != target_id].head(max_history)

                context_ids    = context["anime_id"].astype(int).tolist()
                context_scores = context["score_norm"].tolist()

                # Skip samples where the user has no context at all
                if not context_ids:
                    continue

                self.samples.append({
                    "user_id":        user_id,
                    "target_id":      target_id,
                    "context_ids":    context_ids,
                    "context_scores": context_scores,
                })

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_id": s["user_id"],
            "target_id": s["target_id"],
            "context_ids": s["context_ids"],
            "context_scores": s["context_scores"],
            "target_text": self.e5_prefix + self.id_to_text.get(s["target_id"], ""),
        }


class TripletDataset(Dataset):
    """
    interactions : cleaned interaction DataFrame
    id_to_text : {anime_id: text_string}
    negatives_per_user : how many (anchor, pos, neg) triples to mine per user
    e5_prefix : multilingual-e5 passage prefix
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        id_to_text: dict[int, str],
        negatives_per_user: int = 5,
        e5_prefix: str = "passage: ",
    ):
        self.id_to_text = id_to_text
        self.e5_prefix  = e5_prefix
        self.triplets: list[dict] = []

        for user_id, group in interactions.groupby("user_id"):
            positives = group[group["score_raw"] >= SCORE_HIGH]
            negatives = group[group["score_raw"] <= SCORE_LOW]
            mid_range = group[(group["score_raw"] >= 5) & (group["score_raw"] <= 7)]

            if len(positives) == 0:
                continue

            # Fallback for negatives: take the lowest-scored items
            if len(negatives) == 0:
                negatives = group.nsmallest(max(1, len(group) // 4), "score_raw")

            # Fallback for anchor: use positives
            anchors = mid_range if len(mid_range) > 0 else positives

            for _ in range(negatives_per_user):
                anchor_row = anchors.sample(1).iloc[0]
                pos_row    = positives.sample(1).iloc[0]
                neg_row    = negatives.sample(1).iloc[0]

                score_gap = float(pos_row["score_raw"] - neg_row["score_raw"]) / 9.0

                self.triplets.append({
                    "anchor_id":   int(anchor_row["anime_id"]),
                    "positive_id": int(pos_row["anime_id"]),
                    "negative_id": int(neg_row["anime_id"]),
                    "score_gap":   score_gap,
                })

        random.shuffle(self.triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        prefix = self.e5_prefix
        return {
            "anchor_text": prefix + self.id_to_text.get(t["anchor_id"],   ""),
            "positive_text": prefix + self.id_to_text.get(t["positive_id"], ""),
            "negative_text": prefix + self.id_to_text.get(t["negative_id"], ""),
            "score_gap": torch.tensor(t["score_gap"], dtype=torch.float),
        }


def collate_user_ratings(batch: list[dict]) -> dict:
    """
    Returns
      user_ids : LongTensor  [B]
      target_ids : LongTensor  [B]
      target_texts : list[str]   length B
      context_ids : LongTensor  [B, L]   (0-padded)
      context_scores : FloatTensor [B, L]   (0-padded)
      context_mask : BoolTensor  [B, L]   True = valid position
    """
    max_len = max(len(s["context_ids"]) for s in batch)

    user_ids = []
    target_ids = []
    target_texts = []
    context_ids = []
    context_scores = []
    context_masks = []

    for s in batch:
        L = len(s["context_ids"])
        pad = max_len - L

        user_ids.append(s["user_id"])
        target_ids.append(s["target_id"])
        target_texts.append(s["target_text"])

        context_ids.append(s["context_ids"]    + [0] * pad)
        context_scores.append(s["context_scores"] + [0.0] * pad)
        context_masks.append([True] * L + [False] * pad)

    return {
        "user_ids": torch.tensor(user_ids,   dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "target_texts": target_texts,
        "context_ids": torch.tensor(context_ids,    dtype=torch.long),
        "context_scores": torch.tensor(context_scores, dtype=torch.float),
        "context_mask": torch.tensor(context_masks,  dtype=torch.bool),
    }


def make_anime_loader(
    anime_df: pd.DataFrame,
    batch_size: int = 256,
    num_workers: int = 2,
    e5_prefix: str = "passage: ",
) -> DataLoader:
    ds = AnimeTextDataset(anime_df, e5_prefix=e5_prefix)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def make_user_ratings_loader(
    interactions: pd.DataFrame,
    id_to_text: dict[int, str],
    batch_size: int = 256,
    max_history: int = 50,
    num_workers: int = 2,
    shuffle: bool = True,
    e5_prefix: str = "passage: ",
    min_positives: int = 2,
) -> DataLoader:
    ds = UserRatingsDataset(
        interactions, id_to_text,
        max_history=max_history, e5_prefix=e5_prefix,
        min_positives=min_positives,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_user_ratings,
    )


def make_triplet_loader(
    interactions: pd.DataFrame,
    id_to_text: dict[int, str],
    batch_size: int = 128,
    negatives_per_user: int = 5,
    num_workers: int = 2,
    e5_prefix: str = "passage: ",
) -> DataLoader:
    ds = TripletDataset(
        interactions, id_to_text,
        negatives_per_user=negatives_per_user,
        e5_prefix=e5_prefix,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
