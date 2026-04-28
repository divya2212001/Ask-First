"""
Ask First - AI Health Reasoning Platform
Updated Production Streamlit Application
Includes:
1. JSON Output Section
3. All Users Dashboard + Total Patterns
4. Premium UI Upgrade
"""

import streamlit as st
import json
import pandas as pd

from core.loader import load_dataset
from core.reasoner import run_analysis


st.set_page_config(
    page_title="Ask First AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
html, body, [class*="css"]{
    background:#020617;
    color:white;
    font-family: Inter, sans-serif;
}

/* Header */
.main-header{
    font-size:46px;
    font-weight:800;
    color:#10b981;
    margin-bottom:0;
}
.sub-header{
    color:#94a3b8;
    font-size:16px;
    margin-top:-6px;
}

/* Cards */
.card{
    background:linear-gradient(145deg,#0f172a,#111827);
    border:1px solid #1e293b;
    padding:22px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 0 15px rgba(0,0,0,.25);
}

.card-number{
    font-size:32px;
    font-weight:800;
    color:white;
}

.card-label{
    color:#94a3b8;
    font-size:14px;
}

/* Pattern */
.pattern-box{
    background:#0b1220;
    border:1px solid #1e293b;
    border-radius:18px;
    padding:22px;
    margin-bottom:18px;
}

.pattern-title{
    font-size:22px;
    font-weight:700;
}

.reason-box{
    background:#111827;
    border-left:4px solid #3b82f6;
    padding:14px;
    border-radius:0 10px 10px 0;
    margin-top:12px;
    color:#d1d5db;
}

/* JSON */
.json-box{
    background:#07111f;
    border:1px solid #1e40af;
    border-radius:16px;
    padding:16px;
    font-size:14px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">Ask First AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-conversation pattern detection with temporal reasoning</div>', unsafe_allow_html=True)
st.divider()


with st.sidebar:
    st.header("Configuration")

    uploaded = st.file_uploader(
        "Upload Dataset",
        type=["json"]
    )

    st.divider()

    st.subheader("About")
    st.caption("""
This system detects hidden health patterns across multiple conversations.

Confidence Factors:
- Repetition
- Temporal Consistency
- Counterfactual Support
- Medical Plausibility
- Dose Response
""")


if uploaded:

    try:
        data = load_dataset(uploaded)
        users = data["users"]
    except Exception as e:
        st.error(f"Dataset Error: {e}")
        st.stop()


    st.subheader("Global Dataset Dashboard")

    total_users = len(users)
    total_sessions = sum(len(u.get("conversations", [])) for u in users)

    all_patterns = 0
    high_patterns = 0

    for u in users:
        try:
            r = run_analysis(u, use_llm=False)
            all_patterns += len(r.patterns)
            high_patterns += sum(
                1 for p in r.patterns if p.confidence_label == "high"
            )
        except:
            pass

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown(f"""
        <div class="card">
            <div class="card-number">{total_users}</div>
            <div class="card-label">Users</div>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown(f"""
        <div class="card">
            <div class="card-number">{total_sessions}</div>
            <div class="card-label">Sessions</div>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown(f"""
        <div class="card">
            <div class="card-number">{all_patterns}</div>
            <div class="card-label">Patterns Found</div>
        </div>
        """, unsafe_allow_html=True)

    with d4:
        st.markdown(f"""
        <div class="card">
            <div class="card-number">{high_patterns}</div>
            <div class="card-label">High Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    labels = [f"{u['user_id']} - {u.get('name','Unknown')}" for u in users]
    selected = st.selectbox("Select User Profile", labels)

    uid = selected.split(" - ")[0]
    user = next(u for u in users if u["user_id"] == uid)

    conversations = user.get("conversations", [])

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (user["user_id"], "User ID"),
        (user.get("name", "N/A"), "Name"),
        (len(conversations), "Sessions"),
        (user.get("age", "?"), "Age")
    ]

    for col, (num, lab) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-number">{num}</div>
                <div class="card-label">{lab}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    if st.button("Run Pattern Analysis", use_container_width=True):

        with st.spinner("Running temporal reasoning engine..."):
            result = run_analysis(user, use_llm=True)

        patterns = result.patterns

        # Summary
        st.subheader("Analysis Summary")

        total = len(patterns)

        avg_conf = round(
            sum(p.confidence_score for p in patterns) / total, 2
        ) if total else 0

        s1, s2, s3 = st.columns(3)
        s1.metric("Patterns", total)
        s2.metric("Average Confidence", avg_conf)
        s3.metric("Sessions", len(conversations))

        st.divider()

        st.subheader("Detected Health Patterns")

        json_output = []

        for i, p in enumerate(patterns, 1):

            conf_color = (
                "#22c55e" if p.confidence_label == "high"
                else "#facc15" if p.confidence_label == "medium"
                else "#ef4444"
            )

            st.markdown(f"""
            <div class="pattern-box">
                <div class="pattern-title">
                    {i}. {p.pattern_title}
                </div>

                <div style="margin-top:8px;color:{conf_color};font-weight:700;">
                    Confidence: {p.confidence_label.upper()} ({p.confidence_score})
                </div>

                <div class="reason-box">
                    <b>Reasoning:</b><br>
                    {p.reasoning_trace}
                </div>
            </div>
            """, unsafe_allow_html=True)

            json_output.append({
                "pattern": p.pattern_title,
                "confidence_score": p.confidence_score,
                "confidence_label": p.confidence_label,
                "reasoning": p.reasoning_trace,
                "sessions": p.evidence.session_ids
            })


        st.subheader("JSON Output")

        json_text = json.dumps(json_output, indent=4)

        st.code(json_text, language="json")

        st.download_button(
            "Download JSON Report",
            data=json_text,
            file_name=f"{uid}_patterns.json",
            mime="application/json"
        )

else:
    st.info("Upload your dataset to begin.")

st.divider()
st.caption("Ask First AI v3.0 | Premium Reasoning Engine")

