"""BERTopic theme extraction over the same Reddit MH corpus.

Pipeline: MiniLM embeddings -> UMAP (dim-reduce) -> HDBSCAN (cluster)
-> c-TF-IDF (per-cluster keyword extraction). c-TF-IDF is what makes
the topics interpretable — it surfaces words that are characteristic
of a cluster vs. the rest of the corpus, not just frequent words.
"""
from __future__ import annotations

import json

import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from src import config


def fit_topic_model(texts: list[str]) -> BERTopic:
    embedder = SentenceTransformer(config.EMBED_MODEL)

    # English stopwords + bigrams; bigrams catch phrases like "panic attack"
    # min_df=2 is permissive enough for small corpora (incl. synthetic
    # smoke tests) while still pruning hapax legomena on real data.
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    # reduce_frequent_words pushes generic words ("feel", "like") down,
    # so distinctive vocabulary surfaces in the topic representation.
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        min_topic_size=config.MIN_TOPIC_SIZE,
        calculate_probabilities=False,
        verbose=True,
    )
    topic_model.fit(texts)
    return topic_model


def main():
    df = pd.concat([
        pd.read_csv(config.TRAIN_CSV),
        pd.read_csv(config.VAL_CSV),
    ], ignore_index=True)
    texts = df["text"].astype(str).tolist()
    print(f"[topic] fitting BERTopic on {len(texts)} posts")

    topic_model = fit_topic_model(texts)

    info = topic_model.get_topic_info()
    print("[topic] discovered topics:")
    print(info.head(20).to_string())

    config.TOPIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # safetensors serialization avoids pickling the embedder weights;
    # loader will re-attach the SentenceTransformer at inference time.
    topic_model.save(
        str(config.TOPIC_MODEL_DIR),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=config.EMBED_MODEL,
    )
    info.to_csv(config.TOPIC_MODEL_DIR / "topic_info.csv", index=False)

    summary = {}
    for topic_id in info["Topic"].tolist():
        if topic_id == -1:  # outlier cluster
            continue
        words = topic_model.get_topic(topic_id)
        summary[int(topic_id)] = [w for w, _ in words[:10]]
    with open(config.TOPIC_MODEL_DIR / "topic_keywords.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[topic] saved -> {config.TOPIC_MODEL_DIR}")


if __name__ == "__main__":
    main()
