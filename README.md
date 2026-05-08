# Mental Health Discourse Analyzer

End-to-end NLP pipeline that classifies the distress severity of a
Reddit-style post and surfaces the discourse themes it belongs to.

- **Classifier:** fine-tuned `roberta-base` (3-class severity: mild /
  moderate / severe), weighted CrossEntropy loss for class imbalance.
- **Theme extraction:** BERTopic (MiniLM embeddings → UMAP → HDBSCAN
  → c-TF-IDF) over the same corpus.
- **Serving:** FastAPI `/predict` endpoint with confidence scores +
  themes; Streamlit dashboard on top.
- **Tracking:** MLflow logs hyperparams, per-class precision/recall/F1,
  and the model artifact for every fine-tuning run.

## Architecture

```
Reddit MH posts
      │
      ├──► RoBERTa fine-tune  ─►  models/roberta_distress/
      │       (weighted loss,
      │        MLflow tracking)
      │
      └──► BERTopic           ─►  models/bertopic/
              (MiniLM + c-TF-IDF)

           ┌──────────────┐
           │  FastAPI     │  /health  /predict  /predict_batch
           │  api/main.py │
           └──────┬───────┘
                  │ HTTP
           ┌──────▼───────┐
           │  Streamlit   │  classify form, theme browser,
           │  dashboard   │  test-set metrics
           └──────────────┘
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the pipeline

```bash
# 1. Pull the Reddit MH corpus, derive severity labels, split.
#    Use --synthetic for an offline smoke test.
python -m src.data_prep
# python -m src.data_prep --synthetic

# 2. Fine-tune RoBERTa. Logs to ./mlruns
python -m src.train_classifier

# 3. Fit BERTopic over the same posts.
python -m src.topic_model

# 4. Inspect runs:
mlflow ui --backend-store-uri ./mlruns
```

## Serve

```bash
# terminal 1
python -m api.main

# terminal 2
streamlit run dashboard/app.py
```

Then open the dashboard in your browser. The API docs are at
`http://localhost:8000/docs`.

## Project layout

```
src/
  config.py              all paths, label maps, hyperparams
  data_prep.py           load -> clean -> label (weak supervision) -> split
  train_classifier.py    RoBERTa fine-tune w/ weighted loss + MLflow
  topic_model.py         BERTopic pipeline
  inference.py           shared predict() used by API + CLI
api/main.py              FastAPI service
dashboard/app.py         Streamlit dashboard
```

## Design notes

- **Weak supervision via subreddit** — there is no public Reddit dataset
  pre-labeled with distress severity. We use the
  `solomonk/reddit_mental_health_posts` corpus (adhd, aspergers,
  depression, ocd, ptsd) and map each subreddit to a 3-level distress
  scale (`SUBREDDIT_TO_SEVERITY` in `src/config.py`): adhd/aspergers ->
  mild, depression/ocd -> moderate, ptsd -> severe. The labels are
  noisy, but the classifier learns to generalize from text content
  rather than memorizing subreddit-name cues (which it never sees).
- **Class imbalance handling** — `compute_class_weight(class_weight="balanced")`
  feeds class weights into a custom `WeightedTrainer.compute_loss`. We
  select the best checkpoint by **macro F1**, not accuracy.
- **Shared `inference.py`** — both FastAPI and any CLI/notebook call the
  same `predict()` function. Models are cached with `@lru_cache(maxsize=1)`
  so each worker loads them once.
- **Lifespan startup** — the API warms model caches on boot; the first
  user request isn't a cold-start.
- **Not a clinical tool.** This is a portfolio NLP project. It is not
  intended for any real-world triage, screening, or crisis detection.

## Stack

`PyTorch` · `HuggingFace Transformers` · `BERTopic` · `sentence-transformers`
· `scikit-learn` · `MLflow` · `FastAPI` · `Streamlit` · `Plotly`
