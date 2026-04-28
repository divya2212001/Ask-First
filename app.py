import streamlit as st
import json
from datetime import datetime

from core.loader import load_dataset
from core.reasoner import run_analysis

st.set_page_config(
    page_title="Ask First AI",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"]{
    background:#020617;
    color:white;
    font-family:Inter,sans-serif;
}

.main-title{
    font-size:48px;
    font-weight:800;
    color:#10b981;
    margin-bottom:0;
}

.sub-title{
    color:#94a3b8;
    margin-top:-8px;
    font-size:15px;
}

.metric-card{
    background:linear-gradient(145deg,#0f172a,#111827);
    border:1px solid #1e293b;
    padding:22px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 0 18px rgba(0,0,0,.25);
}

.metric-number{
    font-size:32px;
    font-weight:800;
}

.metric-label{
    font-size:13px;
    color:#94a3b8;
}

.pattern-box{
    background:#0b1220;
    border:1px solid #1e293b;
    border-radius:18px;
    padding:20px;
    margin-bottom:18px;
}

section[data-testid="stSidebar"]{
    background:#0f172a;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Ask First AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Cross-conversation pattern detection with temporal reasoning</div>', unsafe_allow_html=True)
st.divider()


def normalize_patterns(patterns):
    cleaned = []
    seen = set()

    blocked = [
        "water triggers stomach",
        "dehydration triggers stomach",
        "screen time triggers stomach",
        "water triggers stomach pain",
        "dehydration triggers stomach pain"
    ]

    replacements = {
        "water triggers headache": "Low hydration triggers headache",
        "dehydration triggers headache": "Low hydration triggers headache",
        "late eating triggers burning": "Late eating triggers acidity",
        "late eating triggers stomach": "Late eating triggers acidity"
    }

    for p in patterns:
        title = p.pattern_title.strip().lower()

        if any(b in title for b in blocked):
            continue

        if title in replacements:
            p.pattern_title = replacements[title]

        key = p.pattern_title.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(p)

    cleaned.sort(key=lambda x: x.confidence_score, reverse=True)
    return cleaned


with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload Dataset", type=["json"])

    st.divider()

    st.subheader("About")
    st.caption("This system identifies hidden health patterns across multiple conversations.")


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

    total_patterns = 0
    high_patterns = 0

    for u in users:
        try:
            r = run_analysis(u, use_llm=False)
            pats = normalize_patterns(r.patterns)
            total_patterns += len(pats)
            high_patterns += sum(1 for p in pats if p.confidence_label == "high")
        except:
            pass

    a, b, c, d = st.columns(4)

    cards = [
        (a, total_users, "Users"),
        (b, total_sessions, "Sessions"),
        (c, total_patterns, "Patterns"),
        (d, high_patterns, "High Confidence")
    ]

    for col, num, label in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    labels = [f"{u['user_id']} - {u.get('name','Unknown')}" for u in users]
    selected = st.selectbox("Select User Profile", labels)

    uid = selected.split(" - ")[0]
    user = next(u for u in users if u["user_id"] == uid)

    conversations = user.get("conversations", [])

    p1, p2, p3, p4 = st.columns(4)

    profile = [
        (p1, user["user_id"], "User ID"),
        (p2, user.get("name", "N/A"), "Name"),
        (p3, len(conversations), "Sessions"),
        (p4, user.get("age", "?"), "Age")
    ]

    for col, num, label in profile:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    if st.button("Run Pattern Analysis", use_container_width=True):

        with st.spinner("Running analysis..."):
            result = run_analysis(user, use_llm=True)

        patterns = normalize_patterns(result.patterns)

        total = len(patterns)
        avg = round(sum(p.confidence_score for p in patterns) / total, 2) if total else 0

        st.subheader("Analysis Summary")

        s1, s2, s3 = st.columns(3)
        s1.metric("Patterns", total)
        s2.metric("Average Confidence", avg)
        s3.metric("Sessions", len(conversations))

        st.divider()

        st.subheader("Detected Health Patterns")

        json_output = []

        if not patterns:
            st.warning("No meaningful patterns detected.")

        for i, p in enumerate(patterns, 1):

            st.markdown(f"""
            <div class="pattern-box">
                <b>{i}. {p.pattern_title}</b><br><br>
                Confidence: {p.confidence_label.upper()} ({p.confidence_score})<br>
                Why: {p.confidence_justification}<br><br>
                Reasoning: {p.reasoning_trace}<br><br>
                {"Medical Note: " + p.medical_latency_note if p.medical_latency_note else ""}
            </div>
            """, unsafe_allow_html=True)

            json_output.append({
                "pattern_id": p.pattern_id,
                "pattern": p.pattern_title,
                "pattern_type": p.pattern_type,
                "confidence_score": p.confidence_score,
                "confidence_label": p.confidence_label,
                "confidence_justification": p.confidence_justification,
                "reasoning": p.reasoning_trace,
                "medical_latency_note": p.medical_latency_note,
                "sessions": p.evidence.session_ids,
                "root_cause": p.root_cause,
                "downstream_effects": p.downstream_effects
            })

        st.divider()

        st.subheader("Structured JSON Output")

        with st.expander("View JSON"):
            st.json(json_output)

        st.download_button(
            "Download JSON Report",
            data=json.dumps(json_output, indent=4),
            file_name=f"{uid}_patterns.json",
            mime="application/json"
        )

else:
    st.info("Upload dataset to begin.")

st.divider()
st.caption(f"Ask First AI v5.0 | {datetime.now().year}")