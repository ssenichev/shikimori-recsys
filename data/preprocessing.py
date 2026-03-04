import ast
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


RATING_ORDER = {
    "g": 0,
    "pg": 1,
    "pg_13": 2,
    "r": 3,
    "r_plus": 4,
    "rx": 5,
}

SEASON_ORDER = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
_SHIKI_TAG_RE = re.compile(r"\[/?[a-z]+=?\d*\]", re.IGNORECASE)
MIN_RATINGS_PER_USER = 2
SCORE_MIN, SCORE_MAX = 1, 10
SCORE_HIGH = 7  # minimum score to be a valid retrieval target


def _safe_literal(value, fallback=None):
    if pd.isna(value):
        return fallback
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return fallback


def _extract_genre_ids(raw) -> list[int]:
    parsed = _safe_literal(raw, [])
    if not isinstance(parsed, list):
        return []
    ids = []
    for item in parsed:
        if isinstance(item, dict) and "id" in item:
            try:
                ids.append(int(item["id"]))
            except (ValueError, TypeError):
                pass
    return sorted(ids)


def _extract_studio_names(raw) -> list[str]:
    parsed = _safe_literal(raw, [])
    if not isinstance(parsed, list):
        return []
    return [s["name"] for s in parsed if isinstance(s, dict) and "name" in s]


def _extract_year(raw) -> Optional[int]:
    parsed = _safe_literal(raw, {})
    if isinstance(parsed, dict):
        y = parsed.get("year")
        if y is not None:
            try:
                return int(y)
            except (ValueError, TypeError):
                pass
    return None


def _clean_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _SHIKI_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_season(raw) -> tuple[Optional[int], Optional[int]]:
    if not isinstance(raw, str):
        return None, None
    parts = raw.split("_")
    if len(parts) != 2:
        return None, None
    season_name, year_str = parts
    try:
        year = int(year_str)
    except ValueError:
        return None, None
    return year, SEASON_ORDER.get(season_name)


def _normalise_score(score: float) -> float:
    return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def _parse_anime_id(value) -> "Optional[int]":
    if pd.isna(value):
        return None
    s = str(value).strip()
    
    if s.lstrip("-").isdigit():
        return int(s)
    
    parsed = _safe_literal(s)
    if isinstance(parsed, dict) and "id" in parsed:
        try:
            return int(parsed["id"])
        except (ValueError, TypeError):
            return None
    
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    log.info("Loading raw CSVs from %s", data_dir)

    anime_df  = pd.read_csv(data_dir / "anime.csv")
    genres_df = pd.read_csv(data_dir / "genres.csv")
    rates_df  = pd.read_csv(data_dir / "users_rates.csv")

    log.info(
        "Raw sizes — anime: %d  genres: %d  interactions: %d",
        len(anime_df), len(genres_df), len(rates_df),
    )
    return anime_df, genres_df, rates_df


def process_anime(anime_df: pd.DataFrame, genres_df: pd.DataFrame) -> pd.DataFrame:
    df = anime_df.copy()

    genre_map: dict[int, str] = {}
    if "id" in genres_df.columns and "name" in genres_df.columns:
        genre_map = dict(zip(genres_df["id"].astype(int), genres_df["name"].astype(str)))

    df["genre_ids"]   = df["genres"].apply(_extract_genre_ids)
    df["genre_names"] = df["genre_ids"].apply(
        lambda ids: [genre_map.get(i, str(i)) for i in ids]
    )
    
    df["studio_names"] = df["studios"].apply(_extract_studio_names)

    season_parsed = df["season"].apply(lambda s: pd.Series(_parse_season(s), index=["air_year", "season_ordinal"]))
    df["air_year"]       = pd.array(season_parsed["air_year"].tolist(),       dtype=pd.Int64Dtype())
    df["season_ordinal"] = pd.array(season_parsed["season_ordinal"].tolist(), dtype=pd.Int64Dtype())

    mask_no_year = df["air_year"].isna()
    if mask_no_year.any():
        fallback_years = df.loc[mask_no_year, "airedOn"].apply(_extract_year)
        df.loc[mask_no_year, "air_year"] = pd.array(fallback_years.tolist(), dtype=pd.Int64Dtype())

    df["rating_ordinal"] = df["rating"].str.lower().map(RATING_ORDER)

    df["desc_clean"] = df["description"].apply(_clean_description)

    df["score_global"] = pd.to_numeric(df["score"], errors="coerce")
    valid_scores = df["score_global"].between(SCORE_MIN, SCORE_MAX, inclusive="both")
    df.loc[valid_scores, "score_global"] = df.loc[valid_scores, "score_global"].apply(
        _normalise_score
    )
    df.loc[~valid_scores, "score_global"] = np.nan

    def _build_text(row) -> str:
        title = str(row.get("name", "") or "")
        russian = str(row.get("russian", "") or "")
        genres = ", ".join(row["genre_names"]) if row["genre_names"] else ""
        synopsis = row["desc_clean"]

        parts = []
        if title:
            parts.append(title)
        if russian and russian.lower() != title.lower():
            parts.append(f"({russian})")
        if genres:
            parts.append(f"Genres: {genres}.")
        if synopsis:
            parts.append(synopsis)

        return " ".join(parts)

    df["text_input"] = df.apply(_build_text, axis=1)

    log.info("Anime processing done. Non-null text_input: %d / %d", df["text_input"].str.len().gt(0).sum(), len(df))
    return df


def process_interactions(
    rates_df: pd.DataFrame,
    anime_df_processed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Two interaction types
    ─────────────────────
    Explicit (score 1-10):
      The user deliberately assigned a rating.  These are used both as
      training context AND as retrieval targets in the temporal split.
      confidence = score_norm * (1 + log1p(rewatches))
      is_explicit = True

    Implicit (score 0, episodes_watched > 0):
      The user watched the anime but did not rate it.  The act of watching
      is a positive signal; completion rate is a proxy for engagement.
      completion_rate = episodes_watched / total_episodes_in_catalogue
        ∈ [0, 1]  (capped at 1.0 for specials / episode-count mismatches)
      score_norm  = completion_rate * IMPLICIT_SCORE_SCALE   (default 0.5)
        so implicit items are never ranked above a score-6 explicit rating,
        preventing the model from confusing "watched" with "loved".
      confidence  = score_norm   (no rewatch boost — rewatches are 0)
      is_explicit = False

    Implicit interactions with 0 episodes watched are dropped — the row
    was created by the list system but the user never started the show.

    Output columns
    ──────────────
    user_id, anime_id, score_raw (0 for implicit), score_norm, rewatches,
    episodes, completion_rate, confidence, is_explicit, created_at
    """
    # How much to down-weight implicit signals relative to explicit ones.
    # 0.5 means a "completed" watch is treated like a score of ~7/10.
    IMPLICIT_SCORE_SCALE = 0.7

    df = rates_df.copy()

    if "anime" in df.columns and "anime_id" not in df.columns:
        df = df.rename(columns={"anime": "anime_id"})

    df["anime_id"] = df["anime_id"].apply(_parse_anime_id)
    before_id = len(df)
    df = df.dropna(subset=["anime_id"]).copy()
    df["anime_id"] = df["anime_id"].astype(int)
    log.info("Dropped %d rows with unparseable anime_id", before_id - len(df))

    df["score"] = pd.to_numeric(df["score"],    errors="coerce").fillna(0)
    df["rewatches"]= pd.to_numeric(df.get("rewatches", 0), errors="coerce").fillna(0).astype(int)
    df["episodes"] = pd.to_numeric(df.get("episodes", 0),  errors="coerce").fillna(0).astype(int)

    explicit_mask = df["score"].between(SCORE_MIN, SCORE_MAX, inclusive="both")
    implicit_mask = (df["score"] == 0) & (df["episodes"] > 0)

    explicit = df[explicit_mask].copy()
    implicit = df[implicit_mask].copy()

    log.info(
        "Interactions — explicit (scored): %d  implicit (watched, unscored): %d  "
        "dropped (score=0, episodes=0): %d",
        len(explicit), len(implicit),
        len(df) - len(explicit) - len(implicit),
    )

    ep_map: dict[int, int] = {}
    if "episodes" in anime_df_processed.columns:
        ep_map = {
            int(row["id"]): int(row["episodes"])
            for _, row in anime_df_processed.iterrows()
            if pd.notna(row["episodes"]) and int(row["episodes"]) > 0
        }

    def _completion_rate(row) -> float:
        total = ep_map.get(int(row["anime_id"]), 0)
        if total == 0:
            return 1.0
        return min(1.0, row["episodes"] / total)

    explicit["score_raw"] = explicit["score"].astype(int)
    explicit["score_norm"] = explicit["score_raw"].apply(_normalise_score)
    explicit["completion_rate"] = 1.0
    explicit["confidence"] = explicit["score_norm"] * (1 + np.log1p(explicit["rewatches"]))
    explicit["is_explicit"] = True

    implicit["score_raw"] = 0
    implicit["completion_rate"] = implicit.apply(_completion_rate, axis=1)
    implicit["score_norm"] = implicit["completion_rate"] * IMPLICIT_SCORE_SCALE
    implicit["confidence"] = implicit["score_norm"]
    implicit["is_explicit"] = False

    # Promote fully-watched but unrated anime to explicit score=7.
    # Rationale: if a user finished (or nearly finished) a show without
    # dropping it, that's a strong positive signal.  We treat it as a 7/10
    # so these items become valid retrieval targets and training positives.
    IMPLICIT_PROMOTE_THRESHOLD = 0.8   # completion_rate cutoff
    IMPLICIT_PROMOTE_SCORE = 7
    promote_mask = implicit["completion_rate"] >= IMPLICIT_PROMOTE_THRESHOLD
    n_promoted = promote_mask.sum()
    if n_promoted > 0:
        implicit.loc[promote_mask, "score_raw"]    = IMPLICIT_PROMOTE_SCORE
        implicit.loc[promote_mask, "score_norm"]   = _normalise_score(IMPLICIT_PROMOTE_SCORE)
        implicit.loc[promote_mask, "confidence"]   = _normalise_score(IMPLICIT_PROMOTE_SCORE)
        implicit.loc[promote_mask, "is_explicit"]  = True
    log.info(
        "Promoted %d implicit interactions (completion >= %.0f%%) to explicit score=%d",
        n_promoted, IMPLICIT_PROMOTE_THRESHOLD * 100, IMPLICIT_PROMOTE_SCORE,
    )

    df = pd.concat([explicit, implicit], ignore_index=True)

    known_ids = set(anime_df_processed["id"].astype(int))
    before = len(df)
    df = df[df["anime_id"].isin(known_ids)].copy()
    log.info("Dropped %d interactions for unknown anime_id", before - len(df))

    interaction_counts = df.groupby("user_id").size()
    active_users = interaction_counts[interaction_counts >= MIN_RATINGS_PER_USER].index
    before = len(df)
    df = df[df["user_id"].isin(active_users)].copy()
    log.info(
        "Removed users with < %d total interactions (explicit+implicit). Kept: %d users  %d rows",
        MIN_RATINGS_PER_USER, df["user_id"].nunique(), len(df),
    )

    if "createdAt" in df.columns:
        df["created_at"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
        n_missing = df["created_at"].isna().sum()
        if n_missing:
            log.warning("%d rows have unparseable createdAt — filling with epoch", n_missing)
        df["created_at"] = df["created_at"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    else:
        log.warning("No createdAt column — temporal split unavailable")
        df["created_at"] = pd.Timestamp("1970-01-01", tz="UTC")

    cols = ["user_id", "anime_id", "score_raw", "score_norm", "rewatches",
            "episodes", "completion_rate", "confidence", "is_explicit", "created_at"]
    df = df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    n_explicit = df["is_explicit"].sum()
    n_implicit = (~df["is_explicit"]).sum()
    log.info(
        "Final — %d rows (%d explicit, %d implicit)  "
        "%d users  %d items",
        len(df), n_explicit, n_implicit,
        df["user_id"].nunique(), df["anime_id"].nunique(),
    )
    return df


def split_interactions(
    interactions: pd.DataFrame,
    val_frac: float = 0.10,   # unused, kept for API compatibility
    test_frac: float = 0.10,  # unused, kept for API compatibility
    random_state: int = 42,   # unused, kept for API compatibility
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = interactions.copy()

    if "created_at" not in df.columns:
        raise ValueError(
            "split_interactions requires a 'created_at' column. "
            "Make sure process_interactions() parsed the createdAt field."
        )

    # Sort each user's history chronologically; anime_id breaks ties deterministically
    df = df.sort_values(["user_id", "created_at", "anime_id"], ascending=[True, True, True])

    # Temporal rank within HIGH-RATED explicit interactions only.
    # Val/test targets must be items the user liked (score >= SCORE_HIGH),
    # so train and eval are aligned on the same objective: retrieve good recs.
    # Implicit (score=0) watches are always kept in train as context signals.
    if "is_explicit" in df.columns:
        explicit_df = df[
            df["is_explicit"] & (df["score_raw"] >= SCORE_HIGH)
        ].copy()
    else:
        explicit_df = df[df["score_raw"] >= SCORE_HIGH].copy()

    explicit_df["_trank"] = explicit_df.groupby("user_id").cumcount(ascending=False)

    # Users need at least MIN_RATINGS_PER_USER+2 high-rated items to contribute
    # to val/test (so training context has at least 2 positives after holding 2 out).
    user_high_counts = explicit_df.groupby("user_id")["_trank"].max() + 1
    eligible = user_high_counts[
        user_high_counts >= MIN_RATINGS_PER_USER + 2
    ].index

    test_idx = explicit_df[(explicit_df["_trank"] == 0) & explicit_df["user_id"].isin(eligible)].index
    val_idx = explicit_df[(explicit_df["_trank"] == 1) & explicit_df["user_id"].isin(eligible)].index

    test_mask = df.index.isin(test_idx)
    val_mask = df.index.isin(val_idx)

    # _trank was added to explicit_df only, not df — just slice df directly
    test = df[test_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)
    train = df[~(test_mask | val_mask)].reset_index(drop=True)

    log.info(
        "Temporal split — train: %d  val: %d  test: %d  "
        "(eligible users: %d / %d)",
        len(train), len(val), len(test),
        len(eligible), df["user_id"].nunique(),
    )

    # Sanity: val/test items must be strictly after all train items for same user
    _check_temporal_leakage(train, val, test)

    return train, val, test


def _check_temporal_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_max = train.groupby("user_id")["created_at"].max().rename("train_max")
    violations = 0

    for split_name, split_df in [("val", val), ("test", test)]:
        merged = split_df[["user_id", "created_at"]].join(
            train_max, on="user_id", how="left"
        )
        # Flag both strict leakage and timestamp ties (val/test at same second as train)
        strict = merged[merged["created_at"] < merged["train_max"]]
        ties   = merged[merged["created_at"] == merged["train_max"]]
        bad = strict
        if not ties.empty:
            log.warning(
                "Timestamp ties: %d %s rows share created_at with user's latest train item "
                "(not leakage but worth noting)",
                len(ties), split_name,
            )
        if not bad.empty:
            violations += len(bad)
            log.warning(
                "Temporal leakage: %d %s rows have created_at BEFORE "
                "user's latest train item (likely duplicate timestamps)",
                len(bad), split_name,
            )

    if violations == 0:
        log.info("Temporal leakage check passed ✓")


def run_preprocessing(
    data_dir: str | Path,
    output_dir: str | Path,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    random_state: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anime_raw, genres_raw, rates_raw = load_raw(data_dir)

    anime_proc = process_anime(anime_raw, genres_raw)
    anime_proc.to_parquet(output_dir / "anime_processed.parquet", index=False)
    log.info("Saved anime_processed.parquet")

    genre_vocab = {}
    if "id" in genres_raw.columns and "name" in genres_raw.columns:
        genre_vocab = {
            str(row["id"]): row["name"]
            for _, row in genres_raw.iterrows()
        }
    with open(output_dir / "genre_vocab.json", "w", encoding="utf-8") as f:
        json.dump(genre_vocab, f, ensure_ascii=False, indent=2)
    log.info("Saved genre_vocab.json  (%d genres)", len(genre_vocab))

    interactions = process_interactions(rates_raw, anime_proc)

    train, val, test = split_interactions(
        interactions, val_frac=val_frac, test_frac=test_frac, random_state=random_state
    )
    train.to_parquet(output_dir / "train_interactions.parquet", index=False)
    val.to_parquet(output_dir / "val_interactions.parquet",   index=False)
    test.to_parquet(output_dir / "test_interactions.parquet",  index=False)
    log.info("Saved train/val/test interaction parquets")

    stats = {
        "n_anime_raw": len(anime_raw),
        "n_anime_processed": len(anime_proc),
        "n_genres": len(genre_vocab),
        "n_interactions_raw": len(rates_raw),
        "n_interactions_clean": len(interactions),
        "n_users": interactions["user_id"].nunique(),
        "n_items": interactions["anime_id"].nunique(),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "score_mean": round(float(interactions["score_raw"].mean()), 3),
        "score_std": round(float(interactions["score_raw"].std()),  3),
        "density_pct": round(
            100 * len(interactions)
            / (interactions["user_id"].nunique() * interactions["anime_id"].nunique()),
            4,
        ),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Dataset statistics:\n%s", json.dumps(stats, indent=2))

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Shikimori ResSys dataset")
    parser.add_argument("--data_dir", default="./raw_data", help="Directory with raw CSVs")
    parser.add_argument("--output_dir", default="./processed_data", help="Output directory")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int,   default=42)
    args = parser.parse_args()

    run_preprocessing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.seed,
    )
