"""Streamlit dashboard.

Calls the FastAPI service rather than loading models directly. That
mirrors a real deployment (separate API + UI containers) and keeps the
dashboard lightweight.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]
TOPIC_KEYWORDS = ROOT / "models" / "bertopic" / "topic_keywords.json"
TEST_REPORT = ROOT / "models" / "test_report.json"

SEVERITY_COLOR = {
    "none": "#4CAF50",
    "mild": "#FFC107",
    "moderate": "#FF9800",
    "severe": "#F44336",
}

st.set_page_config(page_title="MH Discourse Analyzer", layout="wide")
st.title("Mental Health Discourse Analyzer")
st.caption("RoBERTa distress classifier + BERTopic theme extraction")

# Sidebar — service health and model report
with st.sidebar:
    st.header("Service")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API online")
        st.json(h)
    except Exception as e:
        st.error(f"API offline at {API_URL}")
        st.caption(str(e))
        st.stop()

    if TEST_REPORT.exists():
        st.header("Held-out test metrics")
        report = json.loads(TEST_REPORT.read_text())
        rows = []
        for label in ["none", "mild", "moderate", "severe"]:
            if label in report:
                rows.append({
                    "label": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            macro = report.get("macro avg", {})
            st.metric("Macro F1", f"{macro.get('f1-score', 0):.3f}")

tab_predict, tab_themes = st.tabs(["Classify", "Discovered Themes"])

with tab_predict:
    st.subheader("Classify a post")
    text = st.text_area(
        "Paste a Reddit-style post:",
        height=160,
        placeholder="e.g., I have not been able to sleep in days and nothing brings me joy anymore...",
    )
    if st.button("Classify", type="primary", disabled=not text.strip()):
        with st.spinner("Calling API..."):
            try:
                r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
                r.raise_for_status()
                pred = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        c1, c2 = st.columns([1, 2])
        with c1:
            color = SEVERITY_COLOR.get(pred["label"], "#888")
            st.markdown(
                f"<div style='padding:1.2rem;border-radius:8px;"
                f"background:{color};color:white;text-align:center;'>"
                f"<div style='font-size:0.85rem;opacity:0.85'>Predicted severity</div>"
                f"<div style='font-size:2rem;font-weight:700;text-transform:uppercase;'>"
                f"{pred['label']}</div>"
                f"<div>Confidence: {pred['confidence']:.1%}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            probs_df = pd.DataFrame(
                [{"label": k, "probability": v} for k, v in pred["probabilities"].items()]
            )
            fig = px.bar(
                probs_df, x="label", y="probability",
                color="label", color_discrete_map=SEVERITY_COLOR,
                title="Class probabilities",
            )
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        themes = pred.get("top_themes", [])
        if themes:
            st.subheader("Closest theme")
            for t in themes:
                if t["topic_id"] == -1:
                    st.info("No strong theme match (outlier).")
                else:
                    st.markdown(f"**Topic {t['topic_id']}** — score {t['score']:.2f}")
                    st.write(", ".join(t["keywords"]))

with tab_themes:
    st.subheader("Discovered themes (BERTopic)")
    if TOPIC_KEYWORDS.exists():
        kw = json.loads(TOPIC_KEYWORDS.read_text())
        rows = [
            {"topic_id": int(tid), "keywords": ", ".join(words)}
            for tid, words in sorted(kw.items(), key=lambda x: int(x[0]))
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No topic model trained yet. Run `python -m src.topic_model`.")
