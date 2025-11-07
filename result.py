# --------------------------- IMPORTS ---------------------------
import streamlit as st
import base64
from pathlib import Path
from io import BytesIO
import plotly.graph_objects as go
import numpy as np
import tempfile
import os

# --- Your custom modules (ensure these files are in the same directory) ---
from scoring import score_resume
from recommendation_engine import analyze_resume

# --------------------------- CSS STYLING ---------------------------
CSS_CODE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Basic setup and font */
html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit's default header and footer */
header[data-testid="stHeader"], footer {
    display: none;
}

/* Styling Streamlit's Containers to act as cards */
[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="element-container"] > .st-emotion-cache-1jicfl2 {
    background-color: #FFFFFF;
    border-radius: 15px;
    padding: 1.5rem;
    height: 300px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* Card titles (subheaders) */
h3 {
    color: #111827 !important;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

h6 {
    color: #6B7280 !important;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 1.25rem;
}

/* Specific styling for the Plotly gauge chart container */
[data-testid="stPlotlyChart"] {
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Blue "pills" for actionable recommendations */
.skill-pill {
    display: inline-block;
    background-color: #3B82F6;
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    font-weight: 500;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
}

/* Custom horizontal bar charts for score breakdown */
.score-item {
    display: flex;
    align-items: center;
    margin-bottom: 1.2rem;
    width: 100%;
}
.score-label {
    width: 110px;
    font-weight: 500;
    color: #4B5563;
    font-size: 0.9rem;
}
.score-bar-container {
    flex-grow: 1;
    height: 8px;
    background-color: #E5E7EB;
    border-radius: 4px;
    margin: 0 1rem;
}
.score-bar {
    height: 100%;
    background-color: #8B5CF6;
    border-radius: 4px;
}
.score-value {
    font-weight: 600;
    min-width: 30px;
    color: #1F2937;
    font-size: 0.9rem;
}

/* Custom cards for recommended learning paths */
.learning-link {
    text-decoration: none;
}
.learning-item {
    display: flex;
    align-items: center;
    padding: 0.75rem;
    background-color: #F9FAFB;
    border-radius: 10px;
    margin-bottom: 1rem;
    transition: background-color 0.2s ease;
}
.learning-item:hover {
    background-color: #F3F4F6;
}
.learning-icon {
    width: 40px;
    height: 40px;
    margin-right: 1rem;
    font-size: 1.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
}
.learning-details {
    display: flex;
    flex-direction: column;
}
.learning-course {
    font-weight: 600;
    color: #111827;
    font-size: 0.95rem;
}
.learning-provider {
    font-size: 0.85rem;
    color: #6B7280;
}
"""
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #3B82F6;
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

def load_css_and_background(css_string, background_path):
    """Injects CSS and a background image into the Streamlit app."""
    st.markdown(f'<style>{css_string}</style>', unsafe_allow_html=True)
    if Path(background_path).exists():
        with open(background_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """, unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image '{background_path}' not found.")

# --------------------------- MAIN APP LOGIC ---------------------------

load_css_and_background(CSS_CODE, "background.png")

uploaded_file_bytes = st.session_state.get("uploaded_resume_bytes")
selected_domain = st.session_state.get("selected_domain")

if uploaded_file_bytes is None or selected_domain is None:
    st.error("⚠️ Resume or domain information not found. Please return to the upload page.")
    st.stop()

uploaded_file = BytesIO(uploaded_file_bytes)

# --- TEMP FILE FIX FOR score_resume & analyze_resume ---
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file_bytes)
    tmp_pdf_path = tmp_file.name

try:
    with st.spinner('Analyzing your resume...'):
        scoring_data = score_resume(tmp_pdf_path, selected_domain)
        analysis_data = analyze_resume(tmp_pdf_path, selected_domain)
finally:
    os.remove(tmp_pdf_path)  # clean up temp file

# --------------------------- SCORE PROCESSING ---------------------------
overall_score = max(0, min(int(float(scoring_data.get('probability', 57))), 100))
scores = scoring_data.get('scores', {})
score_breakdown = {
    "Skills": scores.get('skills', 0),
    "Projects": scores.get('projects', 0),
    "Certifications": scores.get('certifications', 0),
    "CGPA": scores.get('cgpa', 0)
}
missing_skills = analysis_data.get("missing_skills", [])
recommendations = analysis_data.get("recommendations", {})

# --------------------------- UI RENDERING ---------------------------
st.markdown('<div class="monitor">', unsafe_allow_html=True)

# --- TOP ROW ---
row1_col1, row1_col2 = st.columns([1,1], gap="large")

with row1_col1:
    with st.container():
        st.markdown("<h3>Overall Score</h3>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            number={'suffix': "%", 'font': {'size': 50, 'color': "#111827"}},
            gauge={
                'axis': {'range': [None, 100], 'visible': False},
                'bar': {'color': "#8B5CF6", 'thickness': 0.8},
                'bgcolor': "#F3F4F6", 'borderwidth': 0,
            }))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with row1_col2:
    with st.container():
        st.markdown("<h3>Actionable Recommendations</h3>", unsafe_allow_html=True)
        st.markdown("<h6>Skills to Add to Your Resume</h6>", unsafe_allow_html=True)
        if missing_skills:
            for skill in missing_skills[:2]:
                st.markdown(f'<div class="skill-pill">{skill}</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 All key skills are present!")

# --- BOTTOM ROW ---
row2_col1, row2_col2 = st.columns([1,1], gap="large")

with row2_col1:
    with st.container():
        st.markdown("<h3>Score Breakdown (Out of 5.0)</h3>", unsafe_allow_html=True)
        if score_breakdown:
            for name, score in score_breakdown.items():
                max_val = 5.0
                display_score = max(0.0, min(float(score), max_val))
                percentage = (display_score / max_val) * 100
                st.markdown(f"""
                    <div class="score-item">
                        <div class="score-label">{name}</div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width: {percentage}%;"></div>
                        </div>
                        <div class="score-value">{display_score:.1f}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Breakdown data is not available.")

with row2_col2:
    with st.container():
        st.markdown("<h3>Recommended Learning Paths</h3>", unsafe_allow_html=True)
        if recommendations:
            provider_icons = {"coursera": "🐍", "udemy": "💻", "default": "📚"}
            for i, (course_name, link) in enumerate(recommendations.items()):
                if i >= 2: break
                provider = "default"
                if "coursera.org" in link: provider = "coursera"
                elif "udemy.com" in link: provider = "udemy"
                st.markdown(f"""
                <a href="{link}" target="_blank" class="learning-link">
                    <div class="learning-item">
                        <div class="learning-icon">{provider_icons.get(provider, '📚')}</div>
                        <div class="learning-details">
                            <div class="learning-course">{course_name}</div>
                            <div class="learning-provider">{provider.capitalize()}</div>
                        </div>
                    </div>
                </a>
                """, unsafe_allow_html=True)
        else:
            st.info("No new learning paths to recommend.")

# --------------------------- BACK BUTTON ---------------------------
if st.button("⬅️ Back to Upload Page"):
    for key in ["uploaded_resume_bytes", "selected_domain"]:
        if key in st.session_state:
            del st.session_state[key]
    st.switch_page("front.py")

st.markdown('</div>', unsafe_allow_html=True)  # Close monitor div
