"""FastAPI inference service.

Endpoints:
  GET  /health         -> liveness
  POST /predict        -> single post -> severity + confidence + themes
  POST /predict_batch  -> list of posts -> list of predictions
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config, inference


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class Theme(BaseModel):
    topic_id: int
    keywords: list[str]
    score: float


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]
    top_themes: list[Theme]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm caches at startup so the first request isn't slow.
    if config.CLASSIFIER_DIR.exists():
        inference._load_classifier()
        inference._load_topic_model()
        print("[api] models warmed")
    else:
        print(f"[api] WARNING: no classifier at {config.CLASSIFIER_DIR}. "
              "Run src/train_classifier.py first.")
    yield


app = FastAPI(
    title="Mental Health Discourse Analyzer",
    description="RoBERTa distress classifier + BERTopic theme extractor.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": config.CLASSIFIER_DIR.exists(),
        "topic_model_loaded": config.TOPIC_MODEL_DIR.exists(),
        "labels": config.LABELS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not config.CLASSIFIER_DIR.exists():
        raise HTTPException(503, "Classifier not trained yet.")
    p = inference.predict(req.text)
    return PredictResponse(**p.__dict__)


@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch(req: BatchRequest):
    if not config.CLASSIFIER_DIR.exists():
        raise HTTPException(503, "Classifier not trained yet.")
    return [PredictResponse(**inference.predict(t).__dict__) for t in req.texts]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config.API_HOST, port=config.API_PORT, reload=False)
