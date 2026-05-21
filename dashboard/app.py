"""Streamlit dashboard.

Calls the FastAPI service rather than loading models directly. That
mirrors a real deployment (separate API + UI containers) and keeps the
dashboard lightweight.

The UI uses a warm, supportive palette intended to feel calm and
non-clinical — this is a research demo about mental-health discourse,
so the styling avoids alarming colours and language.
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

# Muted, supportive severity palette. Severity still needs to be
# visually distinct, but we avoid harsh green/red — softer sage, amber
# and rose read as "gentle caution" rather than "alarm".
SEVERITY_COLOR = {
    "mild": "#8FB996",      # calm green
    "moderate": "#E0A458",  # warm amber
    "severe": "#C97B84",    # muted rose
}

# Encouraging, plain-language framing for each severity level.
SEVERITY_NOTE = {
    "mild": "The post suggests some everyday strain.",
    "moderate": "The post suggests a notable level of distress.",
    "severe": "The post suggests a high level of distress — support matters here.",
}

st.set_page_config(
    page_title="MH Discourse Analyzer",
    page_icon="\U0001F49C",  # purple heart
    layout="wide",
)


def inject_css() -> None:
    """Warm, supportive theme — soft lavender/blue palette, rounded cards."""
    st.markdown(
        """
        <style>
          .stApp {
            background: linear-gradient(180deg, #F6F5FC 0%, #EEF1F8 100%);
          }
          /* Headings in a calm lavender-ink tone */
          h1, h2, h3 { color: #4A4470; }
          /* Soft, rounded primary button */
          .stButton > button {
            background: #6B5B95;
            color: #FFFFFF;
            border: none;
            border-radius: 12px;
            padding: 0.5rem 1.4rem;
            font-weight: 600;
          }
          .stButton > button:hover {
            background: #574A7D;
            color: #FFFFFF;
          }
          /* Rounded, gently shadowed text area */
          .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #D8D4EC;
            background: #FFFFFF;
          }
          /* Tabs with a soft underline accent */
          .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
          .stTabs [aria-selected="true"] { color: #6B5B95; }
          /* Reusable soft card */
          .mh-card {
            background: #FFFFFF;
            border: 1px solid #E4E1F2;
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(75, 68, 112, 0.06);
          }
          .mh-banner {
            background: #EDEAFA;
            border: 1px solid #D8D4EC;
            border-left: 5px solid #6B5B95;
            border-radius: 12px;
            padding: 0.9rem 1.2rem;
            color: #4A4470;
            font-size: 0.92rem;
            line-height: 1.5;
          }
          .mh-disclaimer {
            color: #6E6A84;
            font-size: 0.85rem;
            line-height: 1.5;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

st.title("\U0001F49C Mental Health Discourse Analyzer")
st.caption(
    "A supportive research demo — RoBERTa distress classifier "
    "+ BERTopic theme extraction"
)

# Crisis-support banner — shown on every view. This is a research demo,
# so we make the limits and the real support routes visible up front.
st.markdown(
    """
    <div class="mh-banner">
      <strong>You matter.</strong> This is a research demo and <em>not</em> a
      diagnostic, clinical, or crisis tool. If you or someone you know is
      struggling, please reach out for real support — in the US you can call
      or text <strong>988</strong>, and you can find a helpline in your country
      at <a href="https://findahelpline.com" target="_blank">findahelpline.com</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

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
        for label in ["mild", "moderate", "severe"]:
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
    st.markdown(
        "<p class='mh-disclaimer'>Paste a Reddit-style post below. The model "
        "estimates how much distress the writing expresses — it is a learning "
        "tool, not a judgement about any person.</p>",
        unsafe_allow_html=True,
    )
    text = st.text_area(
        "Post text",
        height=160,
        label_visibility="collapsed",
        placeholder="e.g., I have not been able to sleep in days and nothing brings me joy anymore...",
    )
    if st.button("Analyze post", type="primary", disabled=not text.strip()):
        with st.spinner("Reading the post..."):
            try:
                r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
                r.raise_for_status()
                pred = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        c1, c2 = st.columns([1, 2])
        with c1:
            color = SEVERITY_COLOR.get(pred["label"], "#6B5B95")
            note = SEVERITY_NOTE.get(pred["label"], "")
            st.markdown(
                f"<div style='padding:1.3rem;border-radius:16px;"
                f"background:{color};color:white;text-align:center;'>"
                f"<div style='font-size:0.85rem;opacity:0.9'>Estimated distress level</div>"
                f"<div style='font-size:2rem;font-weight:700;text-transform:uppercase;'>"
                f"{pred['label']}</div>"
                f"<div>Confidence: {pred['confidence']:.1%}</div></div>",
                unsafe_allow_html=True,
            )
            if note:
                st.markdown(
                    f"<p class='mh-disclaimer' style='margin-top:0.6rem'>{note}</p>",
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
            fig.update_layout(
                showlegend=False,
                yaxis_range=[0, 1],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#4A4470",
            )
            st.plotly_chart(fig, use_container_width=True)

        themes = pred.get("top_themes", [])
        if themes:
            st.subheader("Closest theme")
            for t in themes:
                if t["topic_id"] == -1:
                    st.info("No strong theme match — this post reads as an outlier.")
                else:
                    st.markdown(f"**Topic {t['topic_id']}** — score {t['score']:.2f}")
                    st.write(", ".join(t["keywords"]))

        st.markdown(
            "<p class='mh-disclaimer' style='margin-top:1rem'>This estimate comes "
            "from a model trained on noisy, weakly-labelled data. It is not a "
            "diagnosis and should not be used to make decisions about anyone's "
            "care.</p>",
            unsafe_allow_html=True,
        )

with tab_themes:
    st.subheader("Discovered themes (BERTopic)")
    st.markdown(
        "<p class='mh-disclaimer'>These themes were discovered automatically by "
        "clustering the discussion corpus — they describe recurring topics in "
        "the data, not categories of people.</p>",
        unsafe_allow_html=True,
    )
    try:
        r = requests.get(f"{API_URL}/topics", timeout=10)
        r.raise_for_status()
        topics = r.json().get("topics", [])
    except Exception as e:
        st.warning(f"Could not load topics from API: {e}")
        topics = []

    if topics:
        rows = [
            {"topic_id": t["topic_id"], "count": t["count"],
             "keywords": ", ".join(t["keywords"])}
            for t in topics
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    elif TOPIC_KEYWORDS.exists():
        # Local-dev fallback when running without the API.
        kw = json.loads(TOPIC_KEYWORDS.read_text())
        rows = [
            {"topic_id": int(tid), "keywords": ", ".join(words)}
            for tid, words in sorted(kw.items(), key=lambda x: int(x[0]))
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No topics available yet.")
