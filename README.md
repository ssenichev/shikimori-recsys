# Shikimori Anime Recommendation System

A three-stage anime recommendation system for the [Shikimori](https://shikimori.one/) platform. Combines **contrastive text encoder fine-tuning**, **Two-Tower retrieval** with hard negative mining, and a **Cross-Encoder reranker** trained on real user-anime interactions from the Shikimori platform.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Preprocessing](#data-preprocessing)
- [Signals: Positives, Negatives, and Targets](#signals-positives-negatives-and-targets)
- [Train/Val/Test Split](#trainvaltest-split)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)

---

## Architecture Overview

```
                            TRAINING PIPELINE
 ============================================================================

  Raw Data (anime.csv, users_rates.csv, genres.csv)
       |
       v
 +-----------------+
 |  Preprocessing  |   Temporal split, text descriptions,
 |                 |   score normalization, implicit promotion
 +-----------------+
       |
       v
 +------------------------------------------+
 | Stage 1: Contrastive Fine-tuning         |
 |  ContrastiveEncoder (triplet loss)       |
 |  (anchor, positive, negative) + margin   |
 |  Encoder: multilingual-e5 + LoRA         |
 +------------------------------------------+
       |  pretrained ItemTower weights
       v
 +------------------------------------------+
 | Stage 2: Two-Tower Retrieval             |
 |                                          |
 |  ItemTower (frozen)  UserTower (trained) |
 |  +-----------+      +----------------+  |
 |  | e5 + LoRA |      | Score proj     |  |
 |  | + CF emb  |      | Transformer    |  |
 |  | + proj    |      |  Encoder (2L)  |  |
 |  | -> 256D   |      | + CF emb       |  |
 |  +-----------+      | -> 256D        |  |
 |       |             +----------------+  |
 |       v                    v            |
 |     item_emb           user_emb         |
 |        \                 /              |
 |    In-batch + hard-negative softmax     |
 +------------------------------------------+
       |  trained two-tower weights
       v
 +------------------------------------------+
 | Stage 3: Cross-Encoder Reranker          |
 |                                          |
 |  Input: [CLS] user_profile [SEP]        |
 |               anime_text   [SEP]        |
 |                                          |
 |  Encoder: BAAI/bge-reranker-v2-m3       |
 |  Head: sigmoid -> relevance score        |
 |  Loss: MSE on normalized scores          |
 +------------------------------------------+


                          INFERENCE PIPELINE
 ============================================================================

  User History [(anime_id, score), ...]
       |
       v
 +--------------------------+
 | UserTower Encoding       |    TransformerEncoder over
 | -> 256D user embedding   |    watched items + scores + CF
 +--------------------------+
       |
       v
 +--------------------------+
 | Two-Tower Retrieval      |    Dot-product similarity
 | Top-100 candidates       |    with precomputed item embeddings
 +--------------------------+
       |
       v
 +--------------------------+
 | Cross-Encoder Reranking  |    Full attention over
 | Top-K final results      |    (user_profile, anime_text) pairs
 +--------------------------+
       |
       v
  Final Recommendations
  [{rank, anime_id, name, reranker_score, genres, ...}]
```

---

## Data Preprocessing

Source: `data/preprocessing.py`

### Raw Data

Three CSV files from the Shikimori platform:

| File | Contents |
|---|---|
| `anime.csv` | Anime metadata: name, russian title, description, genres, studios, season, rating, episodes, score |
| `genres.csv` | Genre ID-to-name mapping |
| `users_rates.csv` | User-anime interactions: user_id, anime_id, score (0-10), episodes watched, rewatches, createdAt timestamp |

### Anime Processing (`process_anime`)

Each anime is transformed into a structured record with a text representation used by the encoder:

1. **Genre extraction**: Raw JSON genre field parsed into `genre_ids` and `genre_names` via the genres lookup table
2. **Studio extraction**: Studio names extracted from JSON
3. **Season parsing**: `"winter_2020"` -> `air_year=2020, season_ordinal=0`; falls back to `airedOn` field if season is missing
4. **Age rating**: Mapped to ordinal (g=0, pg=1, pg_13=2, r=3, r_plus=4, rx=5)
5. **Description cleaning**: Shikimori BBCode tags (`[b]`, `[url=...]`, etc.) stripped via regex
6. **Global score normalization**: Original 1-10 score normalized to [0, 1] via `(score - 1) / 9`
7. **Text representation** (`text_input`): Concatenation of title, Russian title (if different), genres, and cleaned synopsis

Example `text_input`:
```
Fullmetal Alchemist: Brotherhood (Стальной алхимик: Братство) Genres: Action, Adventure, Drama, Fantasy. Two brothers search for a Philosopher's Stone after an pointless attempt at human transmutation...
```

### Interaction Processing (`process_interactions`)

Every user-anime interaction is classified as **explicit** or **implicit** and assigned derived features:

#### Explicit Interactions (score 1-10)

The user deliberately assigned a rating. These serve as both training context and potential retrieval targets.

| Field | Derivation |
|---|---|
| `score_raw` | Original 1-10 integer |
| `score_norm` | `(score_raw - 1) / 9` -> [0, 1] |
| `completion_rate` | Always 1.0 |
| `confidence` | `score_norm * (1 + log1p(rewatches))` |
| `is_explicit` | `True` |

#### Implicit Interactions (score = 0, episodes > 0)

The user watched but did not rate the anime. The act of watching is a positive signal; completion rate proxies engagement.

| Field | Derivation |
|---|---|
| `score_raw` | 0 |
| `completion_rate` | `min(1.0, episodes_watched / total_episodes)` |
| `score_norm` | `completion_rate * 0.7` (IMPLICIT_SCORE_SCALE) |
| `confidence` | Same as `score_norm` (no rewatch boost) |
| `is_explicit` | `False` |

The `IMPLICIT_SCORE_SCALE = 0.7` ensures implicit items are never ranked above a score-6 explicit rating, preventing the model from confusing "watched" with "loved".

#### Implicit Promotion

Fully-watched but unrated anime are promoted to explicit score=7:

```
If completion_rate >= 0.8:
    score_raw    = 7
    score_norm   = (7 - 1) / 9 = 0.667
    is_explicit  = True
```

**Rationale**: If a user finished (or nearly finished) a show without dropping it, that is a strong positive signal. These promoted items become valid retrieval targets and training positives, significantly increasing the number of high-rated interactions available for training.

#### Dropped Interactions

- `score = 0` AND `episodes = 0`: The row was created by the list system but the user never started the show. **Dropped entirely**.
- Interactions referencing anime IDs not in the processed anime catalog: **Dropped**.
- Users with fewer than 2 total interactions (explicit + implicit): **Dropped**.

---

## Signals: Positives, Negatives, and Targets

### What is a "positive"?

An anime with `score_raw >= 7` (SCORE_HIGH) AND `is_explicit = True`. This includes:
- Anime the user explicitly rated 7, 8, 9, or 10
- Formerly-implicit anime promoted via the completion heuristic (score=7)

### What is a "negative"?

Depends on the training stage:

| Stage | Negative Definition |
|---|---|
| **Stage 1 (Triplet)** | Explicitly scored items with `score_raw <= 4` (SCORE_LOW). Implicit interactions (`score_raw = 0`) are excluded to avoid using "unwatched" as negative signal. Fallback: lowest-scored explicit items if no score <= 4 exists. |
| **Stage 2 (In-batch + Hard)** | Other users' target items in the same batch (in-batch negatives) + top-K highest-scoring non-target items from the full catalog (hard negatives mined via dot-product). |
| **Stage 3 (Reranker)** | Random unseen anime (items the user has never interacted with). 3 negatives per user by default, scored at 0.0. |

### What is the retrieval target?

**A single anime that the user rated highly (score >= 7), held out as the most recent such item in their chronological history.**

During evaluation, we ask: given a user's watch history (everything except the held-out item), can the model rank the held-out positive item highly among all ~10K+ candidate anime?

This is a **single-target retrieval** task. Each user contributes exactly one target item to val/test.

---

## Train/Val/Test Split

Source: `split_interactions` in `data/preprocessing.py`

The split is **temporal leave-last-K-out on high-rated items only**:

1. Sort each user's interactions chronologically by `created_at` (with `anime_id` as a deterministic tiebreaker)
2. Among a user's explicit high-rated items (score >= 7), rank them in reverse chronological order
3. **Test target**: the most recent high-rated item (temporal rank 0)
4. **Val target**: the second most recent high-rated item (temporal rank 1)
5. **Train set**: everything else (all explicit ratings, all implicit watches, all low-rated items)
6. **Eligibility**: only users with >= 4 high-rated items contribute to val/test (so at least 2 positives remain in training context after holding out 2)

A temporal leakage check verifies that no val/test item has a timestamp earlier than the user's latest training item.

```
User timeline:  [item_A (5/10)] [item_B (8/10)] [item_C (3/10)] [item_D (9/10)] [item_E (7/10)]
                 \_______________ train ________________/         \_ val target _/ \_ test target _/
                                                                  (score >= 7)     (score >= 7)
```

Items with score < 7 are never held out as targets, even if they are the user's most recent interaction. This ensures **train and eval are aligned on the same objective**: retrieve anime the user will rate highly.

---

## Model Architecture

### Stage 1: ContrastiveEncoder

Source: `model/architecture.py`

A triplet-loss fine-tuning stage for the text encoder.

**Encoder**: `intfloat/multilingual-e5-base` (or `multilingual-e5-large`) with LoRA adaptation on query/value projection layers.

**Architecture**:
```
Input text (with "passage: " prefix)
    -> XLM-RoBERTa encoder (frozen base + LoRA adapters)
    -> Mean pooling over token embeddings
    -> Dropout + Linear(hidden_dim, proj_dim) + LayerNorm
    -> L2 normalize
    -> 256D unit-norm embedding
```

**LoRA details**: Applied to `query` and `value` linear layers in every attention block. Base weights frozen, only low-rank adapters trained. Default: rank=8, alpha=16, dropout=0.05. Initialization: A with Kaiming uniform, B with zeros (so delta-W starts at 0).

### Stage 2: TwoTowerModel

Two independent towers producing embeddings in the same 256D space.

#### ItemTower

Same architecture as ContrastiveEncoder's tower. Initialized from Stage 1 weights and **frozen during Stage 2** (no gradient updates). Encodes anime text to a 256D unit-norm embedding.

Optional CF fusion: `fused = text_emb + sigmoid(cf_gate) * item_cf_emb[item_idx]`, then re-normalized.

#### UserTower

Aggregates a variable-length watch history into a single 256D user embedding:

```
For each context item:
    item_embedding (from frozen ItemTower cache) + score_proj(score_norm)
        -> [B, L, D] score-conditioned sequence

TransformerEncoder(n_layers=2, nhead=4, dim_feedforward=D*4, norm_first=True)
    -> Masked mean pooling over valid positions
    -> Dropout + Linear(D, D) + LayerNorm
    -> L2 normalize
    -> 256D unit-norm user embedding
```

The `score_proj` is a `Linear(1, proj_dim)` that projects the scalar score into embedding space, allowing the model to condition attention on how much the user liked each item.

Optional CF fusion: `fused = user_emb + sigmoid(cf_gate) * user_cf_emb[user_idx]`, then re-normalized.

#### CF Embeddings

Both towers can optionally incorporate collaborative filtering embeddings:
- `item_cf_emb`: `nn.Embedding(n_items, proj_dim)` — learnable per-item vector
- `user_cf_emb`: `nn.Embedding(n_users, proj_dim)` — learnable per-user vector
- `cf_gate`: Shared scalar parameter initialized at -2.2, so `sigmoid(cf_gate) ~ 0.1`. Controls how much CF signal blends with text embeddings.

During Stage 2, `item_cf_emb` and `cf_gate` are frozen (part of the item-side cache). Only `user_cf_emb` is trained.

### Stage 3: CrossEncoderReranker

Source: `model/reranker.py`

A cross-encoder that scores (user_profile, anime_text) pairs jointly.

**Default encoder**: `BAAI/bge-reranker-v2-m3` with pretrained reranker head (`AutoModelForSequenceClassification`).

**Input format**:
```
[CLS] Enjoyed: Attack on Titan (10), Steins;Gate (9) | Also watched: ... | Disliked: ... [SEP] <anime_text> [SEP]
```

The user profile is a natural-language summary of their taste (top 8 liked, top 3 disliked, top 5 implicitly watched), not the raw interaction sequence. This keeps the input compact and within the 512-token limit.

**Output**: `sigmoid(logits)` -> relevance score in [0, 1].

---

## Training Pipeline

Source: `model/train.py`

### Stage 1: Contrastive Fine-tuning

**Goal**: Teach the text encoder to place semantically similar anime closer in embedding space, guided by user preferences.

**Dataset**: `TripletDataset` — for each user, samples `(anchor, positive, negative)` triplets:
- **Anchor**: Mid-range item (score 5-7) or positive if no mid-range exists
- **Positive**: High-rated item (score >= 7)
- **Negative**: Low-rated item (score <= 4), explicitly scored only (implicit items excluded)

**Data format per sample**:
```python
{
    "anchor_text":   "passage: Fullmetal Alchemist: Brotherhood Genres: Action...",
    "positive_text": "passage: Steins;Gate Genres: Drama, Sci-Fi...",
    "negative_text": "passage: Some Disliked Anime Genres: ...",
    "score_gap":     tensor(0.667),  # (pos_score - neg_score) / 9
}
```

All texts are prefixed with `"passage: "` per multilingual-e5 convention.

**Loss**: Triplet margin loss with adaptive margin:
```
d_ap = 1 - cosine_sim(anchor, positive)
d_an = 1 - cosine_sim(anchor, negative)
margin = base_margin * score_gap
loss = max(0, d_ap - d_an + margin)
```

The `score_gap`-scaled margin means a (9/10 vs 2/10) triplet gets a larger margin than (7/10 vs 5/10).

**Training details**: AdamW optimizer, linear warmup + cosine annealing, gradient accumulation (effective batch = 32 * 4 = 128), fp16 autocast, gradient clipping at 1.0. Default 7 epochs.

### Stage 2: Two-Tower Training

**Goal**: Train the UserTower to produce user embeddings that retrieve the user's next liked anime via dot-product with precomputed item embeddings.

**Key design decision**: **ItemTower is frozen** during Stage 2. The item embedding cache is built once and remains valid for the entire training run. This eliminates the stale-cache problem where context embeddings (from cache) and target embeddings (from live model) would drift into different spaces.

**Frozen parameters**: All ItemTower weights, `item_cf_emb`, `cf_gate`.
**Trained parameters**: UserTower (TransformerEncoder, score_proj, out_proj), `user_cf_emb`.

**Dataset**: `UserRatingsDataset` — for each user's high-rated item, builds a training sample:

```python
{
    "user_id":        42,
    "target_id":      5114,                    # anime to retrieve
    "context_ids":    [1535, 9253, 11061, ...], # previously watched anime IDs
    "context_scores": [0.78, 0.89, 0.89, ...], # normalized scores
    "target_text":    "passage: Fullmetal Alchemist: Brotherhood ...",
}
```

Context is strictly **temporally ordered** — only items with `created_at < target.created_at` are included, up to `max_history=50` most recent. Both explicit and implicit interactions appear in context; only explicit high-rated items serve as targets.

**Collation** (`collate_user_ratings`): Pads variable-length context to batch max, producing:
```
user_ids:       LongTensor  [B]
target_ids:     LongTensor  [B]
target_texts:   list[str]   length B
context_ids:    LongTensor  [B, L]   (0-padded)
context_scores: FloatTensor [B, L]   (0-padded)
context_mask:   BoolTensor  [B, L]   (True = valid position)
```

**Training loop**:

1. Look up context item embeddings from the frozen cache: `item_matrix[context_ids]` -> `[B, L, D]`
2. Look up target embeddings from the frozen cache: `item_matrix[target_cache_idxs]` -> `[B, D]`
3. Encode users: `model.encode_user(context_embs, scores, mask)` -> `[B, D]`
4. Compute in-batch logits: `user_embs @ target_embs.T / temperature` -> `[B, B]`
5. Mine hard negatives: `user_embs @ item_matrix.T` -> mask in-batch targets -> top-K -> `[B, K]` logits
6. Concatenate: `[B, B+K]` logits, labels = `arange(B)`
7. Loss: cross-entropy (sampled softmax)

**Hard negative mining**: Each step computes the full `user_embs @ item_matrix.T` (detached user embeddings), masks out in-batch target indices, and selects the top-K (default 128) highest-scoring non-target items as hard negatives. This forces the model to discriminate between the true target and the most confusable alternatives.

**Training details**: AdamW (only trainable params), linear warmup + cosine annealing, gradient accumulation (effective batch = 64 * 2 = 128), fp16 autocast, gradient clipping at 1.0. Default 10 epochs. Best checkpoint selected by NDCG@10 on validation set.

### Stage 3: Cross-Encoder Reranker Training

**Goal**: Train a cross-encoder to re-score the top-100 candidates from the two-tower retriever using joint attention over user profile and anime text.

**Dataset**: `RerankerDataset` — for each user, every explicit interaction becomes a sample:

```python
{
    "profile_text": "Enjoyed: Attack on Titan (10), Steins;Gate (9) | Disliked: ...",
    "anime_text":   "passage: <anime description>",
    "score_norm":   0.889,   # target: normalized user score
}
```

Plus `negatives_per_user=3` random unseen anime scored at 0.0.

**Collation**: The tokenizer encodes `(profile_text, anime_text)` as a sentence pair with `[SEP]` separator.

**Loss**: MSE regression between `sigmoid(model_output)` and `score_norm`.

**Training details**: AdamW, linear warmup + cosine annealing, gradient accumulation, gradient checkpointing. Default 3 epochs.

---

## Evaluation

Source: `model/metrics.py`

### Metrics

All metrics operate on **ranks** — the position of the held-out target item in the model's ranked list of all catalog items.

| Metric | Formula | Interpretation |
|---|---|---|
| **HR@k** (Hit Rate) | `mean(rank <= k)` | Fraction of users whose target is in the top-k |
| **NDCG@k** | `mean(1/log2(rank+1) if rank <= k else 0)` | Discounted cumulative gain; rewards higher positions more |
| **MRR** (Mean Reciprocal Rank) | `mean(1/rank)` | Average of inverse ranks; heavily rewards rank=1 |

### Retrieval Evaluation (`evaluate_retrieval`)

Used for Stage 2 validation/test:

1. Build user embeddings from context (train history excluding the target)
2. Build item embedding matrix (full catalog)
3. Compute `user_embs @ item_matrix.T` -> cosine similarity scores
4. **Mask seen items**: Set scores of all items in the user's training history to `-1e4` before ranking. This prevents inflated metrics from re-recommending already-watched anime.
5. Compute rank of the target item in the sorted score list
6. Report HR@k, NDCG@k, MRR for k in {5, 10, 20}

### Reranker Evaluation (`evaluate_reranker`)

Used for Stage 3 validation/test:

1. For each user, run the two-tower retriever to get top-100 candidates
2. Check if the target is in the candidate set (`retrieval_recall`)
3. If yes, run the cross-encoder reranker on all 100 candidates
4. Compute the target's rank in the reranked list
5. If the target was not retrieved, rank = infinity (miss)

Reports the same metrics plus `retrieval_recall` (fraction of users whose target appears in the top-100 candidates).

### Train/Eval Alignment

Both training and evaluation target the same signal: **anime the user rated >= 7**. Val/test holdouts are filtered to `score_raw >= SCORE_HIGH` during the temporal split, matching the training objective of retrieving high-rated items.

---

## Project Structure

```
shikimori-recsys/
  data/
    preprocessing.py   # Data loading, cleaning, implicit promotion, temporal split
    dataset.py         # PyTorch Dataset/DataLoader: TripletDataset, UserRatingsDataset, RerankerDataset
  model/
    architecture.py    # ItemTower, UserTower, TwoTowerModel, ContrastiveEncoder, LoRA
    reranker.py        # CrossEncoderReranker, TwoTowerWithReranker (inference wrapper)
    metrics.py         # HR@k, NDCG@k, MRR, evaluate_retrieval, evaluate_reranker
    train.py           # 3-stage training pipeline, HPO with Optuna
  train.ipynb          # End-to-end training and inference notebook
```

### Key Components

| Component | Model | Purpose |
|---|---|---|
| **ItemTower** | `intfloat/multilingual-e5-base` + LoRA + CF | Encodes anime text -> 256D embedding |
| **UserTower** | TransformerEncoder (2 layers, 4 heads) + CF | Aggregates user history -> 256D embedding |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Scores (user_profile, anime) pairs jointly |

---

## Installation & Usage

### Requirements

```bash
pip install torch transformers pandas numpy scikit-learn optuna
```

### Data

The dataset is hosted on HuggingFace: [`kdduha/shikimori-recsys`](https://huggingface.co/datasets/kdduha/shikimori-recsys)

```python
import os

os.makedirs("raw_data", exist_ok=True)
BASE_URL = "https://huggingface.co/datasets/kdduha/shikimori-recsys/resolve/main"

for fname in ["anime.csv", "genres.csv", "users_rates.csv"]:
    os.system(f'wget -q "{BASE_URL}/{fname}" -O "raw_data/{fname}"')
```

### Preprocessing

```python
from data.preprocessing import run_preprocessing

stats = run_preprocessing(data_dir="raw_data", output_dir="processed_data")
# Creates: anime_processed.parquet, train/val/test_interactions.parquet, genre_vocab.json, stats.json
```

### Training (Full Pipeline)

```python
import pathlib, torch
from transformers import AutoTokenizer
from model.architecture import ContrastiveEncoder, TwoTowerModel
from model.reranker import CrossEncoderReranker, train_stage3
from model.train import train_stage1, train_stage2, build_cf_mappings, DEFAULT_CFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = pathlib.Path("checkpoints/")
output_dir.mkdir(exist_ok=True)

ENCODER = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(ENCODER)
cfg = {**DEFAULT_CFG}

# Stage 1
s1_model = ContrastiveEncoder(
    encoder_name=ENCODER, proj_dim=cfg["proj_dim"],
    dropout=cfg["dropout"], freeze_mode=cfg["freeze_mode"],
    lora_rank=cfg["lora_rank"],
)
s1_model = train_stage1(
    s1_model, tokenizer, train_df, val_df, anime_df, cfg, output_dir, device,
)
torch.save(s1_model.tower.state_dict(), output_dir / "stage1_encoder.pt")

# Stage 2
item_id_to_cf_idx, user_id_to_cf_idx, n_items_cf, n_users_cf = build_cf_mappings(anime_df, train_df)
model = TwoTowerModel(
    encoder_name=ENCODER, proj_dim=cfg["proj_dim"], nhead=cfg["nhead"],
    temperature=cfg["temperature"], dropout=cfg["dropout"],
    freeze_mode=cfg["freeze_mode"], lora_rank=cfg["lora_rank"],
    n_items=n_items_cf, n_users=n_users_cf, cf_dim=cfg["cf_dim"],
    user_tower_layers=cfg.get("user_tower_layers", 2),
)
model.item_tower.load_state_dict(
    torch.load(output_dir / "stage1_encoder.pt", map_location="cpu"), strict=False,
)
model, best_val = train_stage2(
    model, tokenizer, train_df, val_df, anime_df, cfg, output_dir, device,
)

# Stage 3
s3_tokenizer = AutoTokenizer.from_pretrained(cfg["s3_encoder"])
reranker = train_stage3(
    CrossEncoderReranker(encoder_name=cfg["s3_encoder"], pretrained_reranker=True),
    s3_tokenizer, train_df, val_df, anime_df, cfg=cfg, output_dir=output_dir, device=device,
)
```

### Inference

```python
from model.reranker import TwoTowerWithReranker

recommender = TwoTowerWithReranker(
    two_tower=model, reranker=reranker,
    tokenizer=tokenizer, reranker_tokenizer=s3_tokenizer,
    anime_df=anime_df, device=device,
    item_id_to_cf_idx=item_id_to_cf_idx,
    user_id_to_cf_idx=user_id_to_cf_idx,
)

# User history: (anime_id, score) — score 1-10 for rated, 0 for watched-but-unrated
recs = recommender.recommend(
    user_history=[
        (5114, 10),   # FMA: Brotherhood
        (9253, 9),    # Steins;Gate
        (11061, 9),   # Hunter x Hunter (2011)
        (37510, 8),   # Mob Psycho 100 II
        (31240, 3),   # Disliked anime
    ],
    top_k=10,
    retrieval_k=100,
    exclude_seen=True,
)

for r in recs:
    print(f"  {r['rank']:2}. {r['name']:<40} (score: {r['reranker_score']:.3f})")
```

### Hyperparameter Optimization

```python
from model.train import run_hpo, run_hpo_reranker

# Stage 2 HPO
best_params = run_hpo(
    train_df, val_df, anime_df, base_cfg=cfg,
    output_dir=output_dir, device=device, n_trials=30, encoder_name=ENCODER,
)

# Stage 3 HPO
best_s3_params = run_hpo_reranker(
    two_tower_model=model, train_df=train_df, val_df=val_df, anime_df=anime_df,
    base_cfg=cfg, output_dir=output_dir, device=device, n_trials=20,
)
```

### Default Configuration

| Parameter | Value | Description |
|---|---|---|
| `proj_dim` | 256 | Embedding dimensionality |
| `user_tower_layers` | 2 | TransformerEncoder depth |
| `nhead` | 4 | Attention heads |
| `s2_batch_size` | 64 | Stage 2 batch size |
| `s2_grad_accum` | 2 | Gradient accumulation steps (effective batch = 128) |
| `s2_hard_neg_k` | 128 | Hard negatives per user per step |
| `cf_dim` | 128 | CF embedding dimension |
| `temperature` | 0.07 | Softmax temperature |
| `lora_rank` | 8 | LoRA adapter rank |
| `s2_max_history` | 50 | Max context items per user |
| `s3_encoder` | `BAAI/bge-reranker-v2-m3` | Reranker base model |
| `retrieval_k` | 100 | Candidates passed to reranker |
