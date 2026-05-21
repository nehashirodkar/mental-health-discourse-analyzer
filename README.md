---
title: Mental Health Discourse Analyzer
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Mental Health Discourse Analyzer

**A fine-tuned RoBERTa distress-severity classifier and a BERTopic theme extractor — trained on a real 7.8k-post Reddit mental-health corpus with weak-supervision labels, tracked with MLflow, served behind a Pydantic-validated FastAPI + a supportive Streamlit dashboard, and deployed on Hugging Face Spaces.**

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-RoBERTa%20fine--tune-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-classifier-FFD21E)
![BERTopic](https://img.shields.io/badge/BERTopic-theme%20extraction-1F9BCF)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics%20%2B%20split-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub%20%2B%20Spaces-FFD21E?logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Hugging%20Face-2496ED?logo=docker&logoColor=white)

[Architecture](#-architecture) · [Demo](#-demo) · [Quickstart](#-quickstart) · [Results](#-measured-results) · [Pipeline](#-pipeline)

---

## ⟡ What it is

**Mental Health Discourse Analyzer** reads a Reddit-style mental-health post and answers two questions: *how much distress does it express*, and *what is it about*. A **fine-tuned `roberta-base`** classifier predicts a 3-level severity — mild / moderate / severe — trained with **class-weighted CrossEntropy** so a ~3:1 class imbalance cannot be won by always predicting the majority class; the best checkpoint is selected by **macro F1**, not accuracy. A **BERTopic** pipeline (MiniLM embeddings → UMAP → HDBSCAN → c-TF-IDF) clusters the same corpus into interpretable discourse themes. Because no public Reddit dataset is pre-labeled with distress severity, labels come from **weak supervision** — each subreddit maps to a severity tier — and the classifier only ever sees post *text*, so it generalizes from content rather than memorizing subreddit-name cues. The whole system is served behind a **Pydantic-validated FastAPI** and a supportive **Streamlit** dashboard, with model caches warmed at startup so the first request is never a cold start.

> **Status: complete + deployed.** End-to-end pipeline (load → clean → weak-label → split → train → evaluate), MLflow experiment tracking, model artifacts on Hugging Face Hub, and a live containerised demo on Hugging Face Spaces — validated end-to-end in production.

## ⟡ Architecture

```
   Reddit-style mental-health post  (Streamlit form)
                │
                ▼
   ┌─────────────────────────┐
   │  FastAPI /predict        │  Pydantic-validated request
   │  FastAPI /topics         │  caches warmed at startup (lifespan)
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐
   │  inference.predict()     │  shared by API + CLI · models @lru_cache'd
   └───────────┬─────────────┘
               │
        ┌──────┴────────────────┐
        ▼                       ▼
   ┌──────────────┐   ┌────────────────────────┐
   │ RoBERTa       │   │ BERTopic                │
   │ distress      │   │ MiniLM → UMAP →         │
   │ classifier    │   │ HDBSCAN → c-TF-IDF      │
   │ (3-class,     │   │ (interpretable theme    │
   │  weighted)    │   │  clusters)              │
   └──────┬───────┘   └───────────┬────────────┘
          │                       │
          ▼                       ▼
   ┌──────────────────────────────────────────────────────┐
   │  response: severity · confidence · class probs ·       │
   │  closest discourse theme + keywords                    │
   │  → Streamlit severity card + probability chart +       │
   │    theme browser                                       │
   └──────────────────────────────────────────────────────┘

   model weights pulled from Hugging Face Hub at container startup
```

## ⟡ Demo

**Live:** https://huggingface.co/spaces/NehaS98/mental-health-discourse-analyzer

**Models:** [classifier](https://huggingface.co/NehaS98/mh-roberta-distress) · [topic model](https://huggingface.co/NehaS98/mh-bertopic)

Paste a Reddit-style post into the dashboard and it returns a predicted distress severity with a confidence value, a full class-probability chart, and the closest discourse theme. A second tab browses every theme BERTopic discovered in the corpus.

| Scenario | What the analyzer does |
|---|---|
| **Classify a post** | RoBERTa scores distress severity — mild / moderate / severe — with confidence + a full class-probability breakdown |
| **Closest theme** | BERTopic maps the post to its nearest discourse cluster and shows that cluster's c-TF-IDF keywords |
| **Outlier post** | When no cluster is a strong match, the post is shown transparently as an outlier rather than forced into a topic |
| **Discovered themes** | The second dashboard tab browses every BERTopic cluster with its keyword summary |

## ⟡ Quickstart

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate                       # Windows
source .venv/bin/activate                    # macOS / Linux
pip install -r requirements.txt

# 2. Build the dataset (Reddit MH corpus → severity labels → stratified split)
python -m src.data_prep
# python -m src.data_prep --synthetic         # offline smoke test

# 3. Train
python -m src.train_classifier                # RoBERTa fine-tune — logs to ./mlruns
python -m src.topic_model                     # BERTopic over the same posts
mlflow ui --backend-store-uri ./mlruns        # inspect runs

# 4. Serve
python -m api.main                            # API on http://localhost:8000
streamlit run dashboard/app.py                # dashboard (separate terminal)
```

> By default, inference pulls model weights from Hugging Face Hub so the deployed Space works out of the box. To serve a **local** checkpoint instead, set `CLASSIFIER_MODEL=models/roberta_distress` and `TOPIC_MODEL=models/bertopic`. The `data/` and `models/` directories are gitignored and recreated by the training scripts on first run.

API docs (Swagger UI) are served at `http://localhost:8000/docs`.

## ⟡ Measured results

Evaluated on a held-out stratified test split of the Reddit mental-health corpus (~7.8k posts; classes are imbalanced roughly 3:1).

- **Strong overall accuracy** — **0.91** on the held-out test set, with a **macro F1 of 0.90** that confirms the model is not coasting on the majority class.
- **The minority class holds up** — **severe-class F1 of 0.86** despite the ~3:1 imbalance, because class-weighted CrossEntropy makes severe-class errors costlier than majority-class errors.
- **Selected for balance, not accuracy** — the best checkpoint is chosen by **macro F1**, so a model that quietly ignores a small class can never win checkpoint selection.
- **Text-only generalization** — the classifier never sees subreddit names, so the score reflects learning from post content rather than memorizing the weak-label source.

| Metric | Score |
|---|---:|
| Accuracy | 0.91 |
| Macro F1 | 0.90 |
| Severe-class F1 | 0.86 |

## ⟡ Pipeline

| Stage | What happens |
|---|---|
| **Acquisition** | Auto-download `solomonk/reddit_mental_health_posts` from HF (adhd, aspergers, depression, ocd, ptsd) |
| **Labeling** | Weak supervision — map each subreddit to a 3-level severity tier; the model trains on text only |
| **Cleaning** | Strip URLs / whitespace, keep posts of 5–400 words, stratified 80 / 10 / 10 train-val-test split |
| **Classifier** | `roberta-base` fine-tune — class-weighted CrossEntropy, `max_len` 256, 3 epochs, warmup 0.1 |
| **Themes** | BERTopic — MiniLM embeddings → UMAP → HDBSCAN → c-TF-IDF over 1–2 grams |
| **Evaluation** | Accuracy + macro precision / recall / F1 + a per-class classification report on the test split |
| **Tracking** | MLflow — hyperparameters, per-epoch + test metrics, class weights, and the model artifact per run |
| **Serving** | Pydantic-validated FastAPI `/predict` · `/predict_batch` · `/topics` + Streamlit dashboard |

## ⟡ Project structure

```
src/
├── config.py            # paths, label maps, hyperparameters, env-var overrides
├── data_prep.py         # load → clean → weak-label (subreddit → severity) → split
├── train_classifier.py  # RoBERTa fine-tune · weighted loss · MLflow tracking
├── topic_model.py       # BERTopic pipeline (MiniLM → UMAP → HDBSCAN → c-TF-IDF)
└── inference.py         # shared predict() used by API + CLI · models @lru_cache'd
api/
└── main.py              # FastAPI service — /health /predict /predict_batch /topics
dashboard/
└── app.py               # supportive Streamlit dashboard (classify + theme browser)
Dockerfile               # Hugging Face Spaces (Docker SDK)
start.sh                 # container entrypoint — FastAPI :8000 + Streamlit :7860
requirements.txt
```

## ⟡ Deployment

| Target | How |
|---|---|
| **Local** | `python -m api.main` + `streamlit run dashboard/app.py` |
| **Docker** | One image runs FastAPI (internal :8000) + Streamlit (:7860) via `start.sh` |
| **Hugging Face Spaces** | Docker SDK Space; model weights pulled from HF Hub at container startup |

## ⟡ Notes

Research / portfolio demonstration — **not a clinical, diagnostic, or crisis-detection tool**, and not intended for any real-world triage or screening. Severity labels are derived by weak supervision from subreddit membership, which is a deliberately noisy proxy: a calm, reflective post can still appear in r/ptsd, and the project pivoted from a 4-class to a 3-class scheme because the public corpus contains no casual / non-MH subreddits to anchor a "none" class. Metrics are reported on a held-out stratified split of the same corpus. If you or someone you know is struggling, please reach out to a qualified professional or a local crisis line.

---
