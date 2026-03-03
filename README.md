# Shikimori Anime Recommendation System

A two-stage anime recommendation system for the [Shikimori](https://shikimori.one/) platform. Combines efficient **Two-Tower retrieval** with an expressive **Cross-Encoder reranker** trained on real user-anime interactions.

## Architecture

```
                            TRAINING PIPELINE
 ============================================================================

  Raw Data (anime.csv, users_rates.csv, genres.csv)
       |
       v
 +-----------------+
 |  Preprocessing  |   Temporal split, text descriptions,
 |                 |   score normalization, implicit/explicit signals
 +-----------------+
       |
       v
 +------------------------------------------+
 | Stage 1: Contrastive Fine-tuning         |
 |  ContrastiveEncoder (triplet loss)       |
 |  (anchor, positive, negative) + margin   |
 |  Encoder: multilingual-e5-base + LoRA    |
 +------------------------------------------+
       |  pretrained ItemTower weights
       v
 +------------------------------------------+
 | Stage 2: Two-Tower Collaborative         |
 |                                          |
 |  ItemTower          UserTower            |
 |  +-----------+      +----------------+  |
 |  | e5-base   |      | Score proj     |  |
 |  | + LoRA    |      | Multi-head     |  |
 |  | + proj    |      |  attention     |  |
 |  | -> 128D   |      | Mean pool      |  |
 |  +-----------+      | -> 128D        |  |
 |       |             +----------------+  |
 |       v                    v            |
 |     item_emb           user_emb         |
 |        \                 /              |
 |         \               /               |
 |    In-batch sampled-softmax loss        |
 +------------------------------------------+
       |  trained two-tower weights
       v
 +------------------------------------------+
 | Stage 3: Cross-Encoder Reranker          |
 |                                          |
 |  Input: [CLS] user_profile [SEP]        |
 |               anime_text   [SEP]        |
 |                                          |
 |  Encoder: DistilBERT multilingual        |
 |  Head: Linear -> sigmoid (regression)    |
 |  Loss: MSE on normalized scores          |
 +------------------------------------------+


                          INFERENCE PIPELINE
 ============================================================================

  User History [(anime_id, score), ...]
       |
       v
 +--------------------------+
 | UserTower Encoding       |    Multi-head attention over
 | -> 128D user embedding   |    watched items + scores
 +--------------------------+
       |
       v
 +--------------------------+
 | Two-Tower Retrieval      |    Dot-product similarity
 | Top-100 candidates       |    with precomputed item embeddings
 | (~10ms)                  |
 +--------------------------+
       |
       v
 +--------------------------+
 | Cross-Encoder Reranking  |    Full attention over
 | Top-K final results      |    (user_profile, anime_text) pairs
 | (~150-200ms for 100)     |    Much more expressive scoring
 +--------------------------+
       |
       v
  Final Recommendations
  [{rank, anime_id, name, reranker_score, genres, ...}]
```

## Project Structure

```
shikimori-recsys/
  data/
    preprocessing.py   # Data loading, cleaning, feature engineering, temporal split
    dataset.py         # PyTorch Dataset/DataLoader classes (triplets, user-item pairs)
  model/
    architecture.py    # ItemTower, UserTower, TwoTowerModel, ContrastiveEncoder, LoRA
    reranker.py        # CrossEncoderReranker, TwoTowerWithReranker (inference wrapper)
    metrics.py         # HR@k, NDCG@k, MRR, retrieval & reranker evaluation
    train.py           # 3-stage training pipeline, HPO with Optuna
  train.ipynb          # End-to-end training and inference notebook
```

## Key Components

| Component | Model | Purpose |
|---|---|---|
| **ItemTower** | `intfloat/multilingual-e5-base` + LoRA | Encodes anime text -> 128D embedding |
| **UserTower** | Multi-head attention (4 heads) | Aggregates user history -> 128D embedding |
| **Reranker** | `distilbert-base-multilingual-cased` | Scores (user_profile, anime) pairs jointly |

## Installation

```bash
pip install torch transformers pandas numpy scikit-learn optuna
```

## Data

The dataset is hosted on HuggingFace: [`kdduha/shikimori-recsys`](https://huggingface.co/datasets/kdduha/shikimori-recsys)

```python
import os

os.makedirs("raw_data", exist_ok=True)
BASE_URL = "https://huggingface.co/datasets/kdduha/shikimori-recsys/resolve/main"

# Download with wget or any HTTP client
for fname in ["anime.csv", "genres.csv", "users_rates.csv"]:
    os.system(f'wget -q "{BASE_URL}/{fname}" -O "raw_data/{fname}"')
```

### Preprocessing

```python
from data.preprocessing import run_preprocessing

stats = run_preprocessing(data_dir="raw_data", output_dir="processed_data")
# Creates: anime_processed.parquet, train/val/test_interactions.parquet
```

## Training

### Full Pipeline (3 stages)

```python
import pathlib
import torch
from transformers import AutoTokenizer
from model.architecture import ContrastiveEncoder, TwoTowerModel
from model.reranker import CrossEncoderReranker, train_stage3
from model.train import train_stage1, train_stage2, DEFAULT_CFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = pathlib.Path("checkpoints/")
output_dir.mkdir(exist_ok=True)

ENCODER = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(ENCODER)
cfg = {**DEFAULT_CFG}

# --- Stage 1: Contrastive fine-tuning of ItemTower ---
s1_model = ContrastiveEncoder(
    encoder_name=ENCODER,
    proj_dim=cfg["proj_dim"],
    dropout=cfg["dropout"],
    freeze_mode=cfg["freeze_mode"],
    lora_rank=cfg["lora_rank"],
)
s1_model = train_stage1(
    s1_model, tokenizer,
    train_df=train_df, val_df=val_df, anime_df=anime_df,
    cfg=cfg, output_dir=output_dir, device=device,
)
torch.save(s1_model.tower.state_dict(), output_dir / "stage1_encoder.pt")

# --- Stage 2: Two-Tower training ---
model = TwoTowerModel(
    encoder_name=ENCODER,
    proj_dim=cfg["proj_dim"],
    nhead=cfg["nhead"],
    temperature=cfg["temperature"],
    dropout=cfg["dropout"],
    freeze_mode=cfg["freeze_mode"],
    lora_rank=cfg["lora_rank"],
)
# Load pretrained item tower from Stage 1
state = torch.load(output_dir / "stage1_encoder.pt", map_location="cpu")
model.item_tower.load_state_dict(state, strict=False)

model, best_val = train_stage2(
    model, tokenizer,
    train_df=train_df, val_df=val_df, anime_df=anime_df,
    cfg=cfg, output_dir=output_dir, device=device,
)

# --- Stage 3: Cross-encoder reranker ---
s3_tokenizer = AutoTokenizer.from_pretrained(cfg["s3_encoder"])
reranker = train_stage3(
    CrossEncoderReranker(encoder_name=cfg["s3_encoder"]),
    s3_tokenizer, train_df, val_df, anime_df,
    cfg=cfg, output_dir=output_dir, device=device,
)
```

### Hyperparameter Optimization

```python
from model.train import run_hpo, run_hpo_reranker

# Stage 2 HPO (Optuna, 20 trials)
best_params = run_hpo(
    train_df=train_df, val_df=val_df, anime_df=anime_df,
    base_cfg=DEFAULT_CFG, output_dir=output_dir,
    device=device, n_trials=20, encoder_name=ENCODER,
)

# Stage 3 HPO
best_s3_params = run_hpo_reranker(
    two_tower_model=model,
    train_df=train_df, val_df=val_df, anime_df=anime_df,
    base_cfg=cfg, output_dir=output_dir,
    device=device, n_trials=20,
)
```

## Inference API

### Loading a Trained Model

```python
import json
import pathlib
import torch
from transformers import AutoTokenizer
from model.architecture import TwoTowerModel
from model.reranker import CrossEncoderReranker, TwoTowerWithReranker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = pathlib.Path("checkpoints/final_model")

# Load config
with open(model_dir / "config.json") as f:
    cfg = json.load(f)

# Load two-tower model
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
model = TwoTowerModel(
    encoder_name="intfloat/multilingual-e5-base",
    proj_dim=cfg["proj_dim"], nhead=cfg["nhead"],
    temperature=cfg["temperature"], dropout=cfg["dropout"],
    freeze_mode=cfg["freeze_mode"], lora_rank=cfg["lora_rank"],
)
model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
model.to(device).eval()

# Load reranker
s3_tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "reranker_tokenizer"))
reranker = CrossEncoderReranker(encoder_name=cfg["s3_encoder"])
reranker.load_state_dict(torch.load(model_dir / "reranker.pt", map_location="cpu"))
reranker.to(device).eval()

# Load anime metadata
anime_df = pd.read_parquet("processed_data/anime_processed.parquet")
```

### Getting Recommendations

```python
# Create the recommender (precomputes item embeddings on init)
recommender = TwoTowerWithReranker(
    two_tower=model,
    reranker=reranker,
    tokenizer=tokenizer,
    reranker_tokenizer=s3_tokenizer,
    anime_df=anime_df,
    device=device,
)

# User history as list of (anime_id, score) tuples
# score: 1-10 for rated anime, 0 for watched-but-unrated
user_history = [
    (5114, 10),    # Fullmetal Alchemist: Brotherhood
    (9253, 9),     # Steins;Gate
    (11061, 9),    # Hunter x Hunter (2011)
    (37510, 8),    # Mob Psycho 100 II
    (31240, 3),    # Disliked anime
]

# Get recommendations
recs = recommender.recommend(
    user_history=user_history,
    top_k=10,          # number of final recommendations
    retrieval_k=100,   # candidates from two-tower retrieval
    exclude_seen=True,
)

# Each recommendation is a dict:
# {
#     "rank": 1,
#     "anime_id": 48569,
#     "name": "Mob Psycho 100 III",
#     "reranker_score": 0.864,
#     "two_tower_rank": 54,          # rank before reranking
#     "genres": ["Action", "Comedy", "Supernatural"],
#     "global_score": 0.87,
# }

for r in recs:
    print(f"  {r['rank']:2}. {r['name']:<40} (score: {r['reranker_score']:.3f})")
```

**Example output:**

```
   1. Mob Psycho 100 III                      (score: 0.864)
   2. Sen to Chihiro no Kamikakushi           (score: 0.844)
   3. One Piece                               (score: 0.842)
   4. Code Geass: Hangyaku no Lelouch R2      (score: 0.832)
   5. Kusuriya no Hitorigoto 2nd Season       (score: 0.831)
   6. Fullmetal Alchemist: Conqueror of Shamballa (score: 0.829)
   7. Takopii no Genzai                       (score: 0.828)
   8. Bocchi the Rock!                        (score: 0.822)
   9. Violet Evergarden                       (score: 0.821)
  10. Hunter x Hunter                         (score: 0.819)
```

## Evaluation Metrics

Evaluated on a temporal hold-out test set (476 users):

| Model | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| Two-Tower only | 0.0210 | 0.0089 | 0.0085 |
| + Cross-Encoder Reranker | 0.0378 | 0.0211 | 0.1517 |

The reranker significantly improves ranking quality - items that the two-tower places at positions 50-100 get promoted to the top-10 when the cross-encoder captures fine-grained preference interactions.

## Dataset Statistics

| Metric | Value |
|---|---|
| Users | 531 |
| Anime items | 3,695 |
| Interactions | 39,345 |
| Explicit (scored) | 27,214 |
| Implicit (watched, unscored) | 12,131 |
| Density | 2.0% |
| Train / Val / Test | 38,393 / 476 / 476 |
