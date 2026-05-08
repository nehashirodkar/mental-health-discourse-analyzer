"""Shared inference: classifier + topic model loaded once, reused.

Used by both FastAPI and any CLI/notebook. Keeping a single predict()
prevents drift between 'what we tested' and 'what we serve'.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config


@dataclass
class Prediction:
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]
    top_themes: list[dict]


@lru_cache(maxsize=1)
def _load_classifier():
    tok = AutoTokenizer.from_pretrained(str(config.CLASSIFIER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.CLASSIFIER_DIR))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


@lru_cache(maxsize=1)
def _load_topic_model():
    try:
        from bertopic import BERTopic
        if not config.TOPIC_MODEL_DIR.exists():
            return None
        return BERTopic.load(str(config.TOPIC_MODEL_DIR))
    except Exception as e:
        print(f"[inference] topic model unavailable: {e}")
        return None


def classify(text: str) -> tuple[int, np.ndarray]:
    tok, model, device = _load_classifier()
    enc = tok(
        text,
        truncation=True,
        max_length=config.MAX_LEN,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return int(probs.argmax()), probs


def themes_for(text: str, top_k: int = 3) -> list[dict]:
    tm = _load_topic_model()
    if tm is None:
        return []
    topics, probs = tm.transform([text])
    topic_id = int(topics[0])
    if topic_id == -1:
        return [{"topic_id": -1, "keywords": [], "score": 0.0}]
    keywords = [w for w, _ in (tm.get_topic(topic_id) or [])][:top_k * 2]
    score = float(probs[0]) if probs is not None and len(probs) else 0.0
    return [{"topic_id": topic_id, "keywords": keywords[:10], "score": score}]


def predict(text: str) -> Prediction:
    label_id, probs = classify(text)
    return Prediction(
        label=config.ID2LABEL[label_id],
        label_id=label_id,
        confidence=float(probs[label_id]),
        probabilities={config.ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        top_themes=themes_for(text),
    )


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "I have not been able to sleep for days and nothing matters"
    print(predict(text))
