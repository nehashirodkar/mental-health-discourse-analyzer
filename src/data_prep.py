"""Load Reddit mental-health posts, derive severity labels, split.

Strategy: each post's subreddit is the weak label source. The 3-level
severity scale in config.SUBREDDIT_TO_SEVERITY maps adhd/aspergers ->
mild, depression/ocd -> moderate, ptsd -> severe. This is imperfect (a
calm, reflective post can still land in r/ptsd) but is a defensible
proxy for a portfolio project — and the classifier learns to generalize
from text content, not subreddit names.
"""
from __future__ import annotations

import argparse
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

URL_RE = re.compile(r"http\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = WS_RE.sub(" ", text).strip()
    return text


def load_from_huggingface() -> pd.DataFrame:
    """Pull a Reddit mental-health dataset from HF.

    We try a few known dataset slugs in order. The user can always
    drop their own CSV at data/reddit_mh_raw.csv with columns
    [text, subreddit] to bypass this.
    """
    from datasets import load_dataset

    candidates = [
        ("solomonk/reddit_mental_health_posts", None),
        ("marmolpen3/mental-health-reddit-comments", None),
    ]
    for name, split in candidates:
        try:
            print(f"[data] trying HF dataset: {name}")
            ds = load_dataset(name, split=split or "train")
            df = ds.to_pandas()
            print(f"[data] loaded {len(df)} rows from {name}")
            return df
        except Exception as e:
            print(f"[data] {name} failed: {e}")
    raise RuntimeError(
        "Could not load any HF dataset. Drop a CSV at "
        f"{config.RAW_CSV} with columns [text, subreddit] and rerun."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Datasets vary in column names — normalize to [text, subreddit]."""
    cols = {c.lower(): c for c in df.columns}

    text_col = next(
        (cols[c] for c in ["text", "body", "selftext", "content", "post", "title"] if c in cols),
        None,
    )
    sub_col = next(
        (cols[c] for c in ["subreddit", "label", "category", "class"] if c in cols),
        None,
    )
    if text_col is None or sub_col is None:
        raise ValueError(
            f"Could not find text/subreddit columns. Got: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "text": df[text_col].astype(str).map(clean_text),
        "subreddit": df[sub_col].astype(str).str.lower().str.strip(),
    })
    return out


def assign_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severity"] = df["subreddit"].map(config.SUBREDDIT_TO_SEVERITY)
    df = df.dropna(subset=["severity"])
    df["severity"] = df["severity"].astype(int)
    return df


def filter_quality(df: pd.DataFrame, min_words: int = 5, max_words: int = 400) -> pd.DataFrame:
    word_counts = df["text"].str.split().str.len()
    keep = (word_counts >= min_words) & (word_counts <= max_words)
    return df.loc[keep].reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = config.SEED):
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df["severity"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["severity"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_synthetic(n_per_class: int = 200) -> pd.DataFrame:
    """Tiny synthetic dataset for smoke-testing the pipeline offline.

    Subreddits here must exist in config.SUBREDDIT_TO_SEVERITY, otherwise
    assign_severity() would drop every synthetic row. They cover all
    three severity tiers: adhd/aspergers -> mild, depression/ocd ->
    moderate, ptsd -> severe.
    """
    samples = {
        "adhd": [
            "I started five different tasks today and finished none of them",
            "Lost my keys again and missed another deadline I forgot about",
            "Reading the same paragraph over and over and it will not stick",
        ],
        "aspergers": [
            "Group conversations move too fast for me to find a way in",
            "The noise in the office was overwhelming and I had to step out",
            "I rehearsed the small talk but it still did not come out right",
        ],
        "depression": [
            "I have not been able to get out of bed for three days now",
            "Nothing brings me joy anymore, I feel completely empty inside",
            "Everything feels gray and pointless, even things I used to love",
        ],
        "ocd": [
            "I checked the front door lock eleven times before I could leave",
            "The intrusive thoughts keep looping and I cannot make them stop",
            "I washed my hands until they were raw before I felt okay",
        ],
        "ptsd": [
            "A car backfired and I was instantly back in that moment, shaking",
            "The nightmares wake me every night and I dread falling asleep",
            "A certain smell sent me into a flashback for the whole afternoon",
        ],
    }
    rows = []
    for sub, texts in samples.items():
        for _ in range(n_per_class):
            for t in texts:
                rows.append({"text": t, "subreddit": sub})
    return pd.DataFrame(rows)


def main(use_synthetic: bool = False, max_rows: int | None = None) -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    if use_synthetic:
        print("[data] using synthetic dataset")
        raw = make_synthetic()
    elif config.RAW_CSV.exists():
        print(f"[data] reading existing {config.RAW_CSV}")
        raw = pd.read_csv(config.RAW_CSV)
    else:
        raw = load_from_huggingface()
        raw = normalize_columns(raw)
        raw.to_csv(config.RAW_CSV, index=False)
        print(f"[data] cached raw to {config.RAW_CSV}")

    if "severity" not in raw.columns:
        if "subreddit" not in raw.columns:
            raw = normalize_columns(raw)
        raw = assign_severity(raw)

    raw = filter_quality(raw)
    print(f"[data] after quality filter: {len(raw)} rows")

    if max_rows is not None and len(raw) > max_rows:
        # Stratified subsample so each class shrinks proportionally.
        raw, _ = train_test_split(
            raw, train_size=max_rows, stratify=raw["severity"],
            random_state=config.SEED,
        )
        raw = raw.reset_index(drop=True)
        print(f"[data] subsampled to {len(raw)} rows (stratified)")

    print("[data] severity distribution:")
    print(raw["severity"].value_counts().sort_index().to_string())

    train, val, test = stratified_split(raw)
    train.to_csv(config.TRAIN_CSV, index=False)
    val.to_csv(config.VAL_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)
    print(f"[data] wrote {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a small synthetic dataset (offline smoke test)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Stratified subsample to this many rows total before split")
    args = parser.parse_args()
    sys.exit(main(use_synthetic=args.synthetic, max_rows=args.max_rows))
