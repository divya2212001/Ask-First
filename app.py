"""
Ask First - AI Health Reasoning Platform
Production Streamlit Application
"""

import streamlit as st
import json

from core.loader import load_dataset
from core.reasoner import run_analysis


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Ask First AI - Health Pattern Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM STYLES
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 800;
        color: #10b981;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 18px;
        color: #9ca3af;
        margin-top: 4px;
    }
    .stat-card {
        background: #0f172a;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-size: 32px;
        font-weight: 800;
        color: white;
    }
    .stat-label {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .pattern-card {
        background: #0b1220;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .pattern-title {
        font-size: 22px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .confidence-high { color: #22c55e; font-weight: 700; }
    .confidence-medium { color: #facc15; font-weight: 700; }
    .confidence-low { color: #ef4444; font-weight: 700; }
    .reasoning-box {
        background: #111827;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
        color: #d1d5db;
    }
    .latency-note {
        background: #1e1b4b;
        border-left: 4px solid #8b5cf6;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        color: #c4b5fd;
        font-size: 14px;
    }
    .evidence-box {
        margin-top: 12px;
        padding: 10px;
        background: #111827;
        border-radius: 8px;
        font-size: 13px;
        color: #94a3b8;
    }
    .strategy-box {
        background: #064e3b;
        border: 1px solid #065f46;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: #a7f3d0;
    }
</style>
""", unsafe_allow_html=True)


# HEADER
st.markdown('<div class="main-header">Ask First AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-conversation pattern detection with temporal reasoning</div>', unsafe_allow_html=True)
st.divider()


# SIDEBAR
with st.sidebar:
    st.header("Configuration")
    
    uploaded = st.file_uploader(
        "Upload dataset (JSON)",
        type=["json"],
        help="Upload askfirst_synthetic_dataset.json or compatible format"
    )
    
    st.divider()
    
    st.subheader("About")
    st.caption("""
    This system detects hidden health patterns across
    multiple conversations using temporal reasoning.
    
    Confidence factors:
    - Repeated evidence
    - Temporal consistency
    - Counterfactual support
    - Medical plausibility
    - Resolution evidence
    - Dose-response
    """)


# MAIN CONTENT
if uploaded:
    
    # Load data
    try:
        data = load_dataset(uploaded)
        users = data["users"]
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
    
    # User selection
    labels = [f"{u['user_id']} - {u.get('name', 'Unknown')}" for u in users]
    selected = st.selectbox("Select User Profile", labels)
    uid = selected.split(" - ")[0]
    
    user = next(u for u in users if u["user_id"] == uid)
    
    # USER STATS
    conversations = user.get("conversations", [])
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-number">{user["user_id"]}</div>
            <div class="stat-label">User ID</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with c2:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-number">{user.get("name", "N/A")}</div>
            <div class="stat-label">Name</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with c3:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-number">{len(conversations)}</div>
            <div class="stat-label">Sessions</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with c4:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-number">{user.get("age", "?")}</div>
            <div class="stat-label">Age</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.divider()
    
    # RUN ANALYSIS
    run_btn = st.button("Run Pattern Analysis", use_container_width=True, type="primary")
    
    if run_btn:
        with st.spinner("Running temporal reasoning pipeline..."):
            result = run_analysis(user, use_llm=True)
        
        # STRATEGY INFO
        st.markdown(f'''
        <div class="strategy-box">
            <b>Reasoning Strategy:</b> {result.reasoning_strategy}<br>
            <b>Chunking Strategy:</b> {result.chunking_strategy}
        </div>
        ''', unsafe_allow_html=True)
        
        # SUMMARY METRICS
        st.subheader("Analysis Summary")
        
        patterns = result.patterns
        total = len(patterns)
        
        if total > 0:
            avg_conf = round(
                sum(p.confidence_score for p in patterns) / total, 2
            )
            high_count = sum(1 for p in patterns if p.confidence_label == "high")
            med_count = sum(1 for p in patterns if p.confidence_label == "medium")
        else:
            avg_conf = 0
            high_count = med_count = 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patterns Found", total)
        m2.metric("Avg Confidence", f"{avg_conf}")
        m3.metric("High Confidence", high_count, delta=f"{high_count}/{total}")
        m4.metric("Medium Confidence", med_count)
        
        st.divider()
        
        # PATTERNS DISPLAY
        st.subheader("Detected Health Patterns")
        
        for i, pattern in enumerate(patterns, 1):
            conf_class = f"confidence-{pattern.confidence_label}"
            status_dot = "●"
            dot_color = "#22c55e" if pattern.confidence_label == "high" else "#facc15" if pattern.confidence_label == "medium" else "#ef4444"
            
            # Medical latency badge
            latency_html = ""
            if pattern.medical_latency_note:
                latency_html = f'''
                <div class="latency-note">
                    <b>Medical Latency:</b> {pattern.medical_latency_note}
                </div>
                '''
            
            # Evidence box
            evidence = pattern.evidence
            dose_str = ""
            if evidence.dose_response_score:
                dose_str = f"<b>Dose-Response Score:</b> {evidence.dose_response_score}<br>"
            
            evidence_html = f'''
            <div class="evidence-box">
                <b>Sessions:</b> {", ".join(evidence.session_ids) or "N/A"}<br>
                <b>Counterfactuals:</b> {len(evidence.counterfactual_sessions)} clean sessions<br>
                <b>Temporal Consistency:</b> {evidence.temporal_consistency}<br>
                {dose_str}
            </div>
            '''
            
            st.markdown(f'''
            <div class="pattern-card">
                <div class="pattern-title">
                    <span style="color: {dot_color};">{status_dot}</span> {i}. {pattern.pattern_title}
                </div>
                <div class="{conf_class}" style="margin-top: 8px;">
                    Confidence: {pattern.confidence_label.upper()} ({pattern.confidence_score})
                </div>
                <div class="reasoning-box">
                    <b>Reasoning:</b><br>{pattern.reasoning_trace}
                </div>
                {latency_html}
                {evidence_html}
            </div>
            ''', unsafe_allow_html=True)
        
        # DOWNLOAD BUTTON - Human-readable report
        def generate_text_report():
            lines = []
            lines.append("=" * 60)
            lines.append("ASK FIRST - HEALTH PATTERN ANALYSIS REPORT")
            lines.append("=" * 60)
            lines.append("")
            lines.append(f"User ID: {result.user_id}")
            lines.append(f"Name: {result.user_name or 'N/A'}")
            lines.append(f"Total Sessions Analyzed: {result.total_sessions}")
            lines.append(f"Patterns Found: {result.patterns_found}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("DETECTED HEALTH PATTERNS")
            lines.append("-" * 60)
            lines.append("")
            
            for i, pattern in enumerate(result.patterns, 1):
                lines.append(f"{i}. {pattern.pattern_title}")
                lines.append(f"   Type: {pattern.pattern_type.replace('_', ' ').title()}")
                lines.append(f"   Confidence: {pattern.confidence_label.upper()} ({pattern.confidence_score})")
                lines.append(f"   Justification: {pattern.confidence_justification}")
                lines.append("")
                lines.append(f"   Reasoning:")
                for reasoning_line in pattern.reasoning_trace.split('. '):
                    if reasoning_line.strip():
                        lines.append(f"      - {reasoning_line.strip()}")
                lines.append("")
                
                if pattern.medical_latency_note:
                    lines.append(f"   Medical Latency: {pattern.medical_latency_note}")
                    lines.append("")
                
                evidence = pattern.evidence
                lines.append(f"   Evidence:")
                lines.append(f"      Sessions involved: {', '.join(evidence.session_ids) or 'N/A'}")
                lines.append(f"      Counterfactual sessions: {len(evidence.counterfactual_sessions)}")
                lines.append(f"      Temporal consistency: {evidence.temporal_consistency}")
                if evidence.dose_response_score:
                    lines.append(f"      Dose-response score: {evidence.dose_response_score}")
                lines.append("")
                
                if pattern.root_cause:
                    lines.append(f"   Root Cause: {pattern.root_cause}")
                if pattern.downstream_effects:
                    lines.append(f"   Downstream Effects: {', '.join(pattern.downstream_effects)}")
                lines.append("")
                lines.append("-" * 40)
                lines.append("")
            
            lines.append("=" * 60)
            lines.append("END OF REPORT")
            lines.append("=" * 60)
            
            return "\n".join(lines)
        
        report_text = generate_text_report()
        
        st.download_button(
            "Download Full Report",
            data=report_text,
            file_name=f"analysis_{result.user_id}.txt",
            mime="text/plain"
        )
        
else:
    st.info("Upload a dataset from the sidebar to begin analysis.")
    
    st.divider()
    st.subheader("Quick Start")
    st.markdown("""
    1. **Upload Dataset**: Use `askfirst_synthetic_dataset.json` or your own data
    2. **Select User**: Choose from detected user profiles
    3. **Run Analysis**: Click the button to detect temporal patterns
    4. **Review Results**: See patterns with confidence scores and reasoning traces
    
    **Supported Pattern Types:**
    - Temporal Sequences (cause -> effect across time)
    - Dose-Response (more trigger = worse symptom)
    - Intervention Response (symptom improves after change)
    - Progressive Decline (cascading symptoms from root cause)
    - Compound Causes (multiple triggers together)
    """)


# FOOTER
st.divider()
st.caption("Ask First AI v2.0 | Production Reasoning Engine")

