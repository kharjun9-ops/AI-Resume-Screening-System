import streamlit as st
import pandas as pd
import os
import time
import tempfile
import concurrent.futures
import hashlib
import json
from src.text_extraction import extract_text_from_file, extract_text_from_bytes
from src.text_preprocessing import preprocess_text, preprocess_texts
from src.feature_extraction import vectorize_text
from src.skills_extraction import extract_skills, HARD_SKILLS, SOFT_SKILLS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import uuid
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

TRAINING_RESUMES_FOLDER = os.path.join("data", "resumes")
TRAINING_JD_PATH = os.path.join("data", "job_descriptions", "job_description.txt")
PARTIAL_LABEL = "exclude"
MANAGERS_FILE = os.path.join("data", "managers.json")

# ── Manager Authentication Helpers ──────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Return a SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _load_managers() -> dict:
    """Load the managers registry from disk."""
    if os.path.exists(MANAGERS_FILE):
        try:
            with open(MANAGERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_managers(managers: dict) -> None:
    """Persist the managers registry to disk."""
    os.makedirs(os.path.dirname(MANAGERS_FILE), exist_ok=True)
    with open(MANAGERS_FILE, "w", encoding="utf-8") as f:
        json.dump(managers, f, indent=2)

def _manager_signup(username: str, password: str) -> tuple[bool, str]:
    """Register a new manager. Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    managers = _load_managers()
    if username.lower() in {u.lower() for u in managers}:
        return False, "This username is already taken."
    managers[username] = _hash_password(password)
    _save_managers(managers)
    return True, "Account created! You can now log in."

def _manager_login(username: str, password: str) -> tuple[bool, str]:
    """Authenticate a manager. Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Please enter both username and password."
    managers = _load_managers()
    stored_hash = managers.get(username)
    if stored_hash is None:
        return False, "Username not found. Please sign up first."
    if stored_hash != _hash_password(password):
        return False, "Incorrect password."
    return True, "Welcome back!"

def render_manager_auth():
    """Render the login / sign-up UI for the Manager Panel.
    Returns True if the user is authenticated, False otherwise.
    """
    if st.session_state.get("manager_logged_in"):
        # Already authenticated — show a small welcome + logout in the sidebar
        st.sidebar.markdown(f"👤 Logged in as **{st.session_state.manager_username}**")
        if st.sidebar.button("Logout", key="manager_logout"):
            st.session_state.manager_logged_in = False
            st.session_state.manager_username = ""
            st.rerun()
        return True

    st.header("🔐 Manager Login")
    st.markdown("Please log in or create an account to access the Manager Panel.")

    auth_tab = st.radio(
        "Choose action",
        ["Login", "Sign Up"],
        horizontal=True,
        key="auth_tab",
        label_visibility="collapsed",
    )

    with st.container(border=True):
        if auth_tab == "Login":
            st.subheader("Login")
            login_user = st.text_input("Username", key="login_user", placeholder="Enter your username")
            login_pass = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            if st.button("Login →", use_container_width=True, type="primary"):
                ok, msg = _manager_login(login_user, login_pass)
                if ok:
                    st.session_state.manager_logged_in = True
                    st.session_state.manager_username = login_user.strip()
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.subheader("Create Account")
            signup_user = st.text_input("Choose a Username", key="signup_user", placeholder="Pick a username")
            signup_pass = st.text_input("Choose a Password", type="password", key="signup_pass", placeholder="Min 4 characters")
            signup_pass2 = st.text_input("Confirm Password", type="password", key="signup_pass2", placeholder="Re-enter password")
            if st.button("Create Account ✨", use_container_width=True, type="primary"):
                if signup_pass != signup_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = _manager_signup(signup_user, signup_pass)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    return False

def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');

            /* Global Background & Mesh Gradient */
            .stApp {
                background-color: #070A13;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(0, 229, 255, 0.05) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(162, 0, 255, 0.05) 0px, transparent 50%);
                background-attachment: fixed;
                font-family: 'Inter', sans-serif;
            }

            /* Typography */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Outfit', sans-serif;
                color: #ffffff;
                letter-spacing: -0.02em;
            }

            p, li, span, label {
                color: #e2e8f0;
            }

            .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div, .stMarkdown li {
                color: #cbd5e1;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: rgba(11, 15, 25, 0.7);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-right: 1px solid rgba(255, 255, 255, 0.05);
            }

            [data-testid="stSidebar"] * {
                color: #e2e8f0;
            }

            /* Block Container */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            /* Custom Hero Section */
            .hero {
                padding: 3rem 2rem;
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2), inset 0 0 0 1px rgba(255, 255, 255, 0.05);
                margin-bottom: 2rem;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            .hero::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(0,229,255,0.05) 0%, transparent 50%);
                z-index: 0;
            }

            .hero-title {
                font-family: 'Outfit', sans-serif;
                font-size: 3.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #ffffff 0%, #00e5ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative;
                z-index: 1;
            }

            .hero-subtitle {
                color: #94a3b8;
                font-size: 1.1rem;
                margin-bottom: 1.5rem;
                position: relative;
                z-index: 1;
                font-weight: 300;
            }

            .pill {
                display: inline-block;
                padding: 0.4rem 1rem;
                border-radius: 999px;
                font-size: 0.85rem;
                background: rgba(0, 229, 255, 0.1);
                border: 1px solid rgba(0, 229, 255, 0.2);
                color: #00e5ff;
                margin: 0 0.4rem;
                position: relative;
                z-index: 1;
                font-weight: 500;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }

            /* Cards & Glassmorphism */
            .section-card {
                background: rgba(17, 22, 37, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                margin-bottom: 1.5rem;
                color: #e2e8f0;
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            
            .section-card:hover {
                border-color: rgba(0, 229, 255, 0.3);
                transform: translateY(-2px);
            }

            /* Buttons */
            .stButton > button, .stDownloadButton > button {
                background: rgba(255, 255, 255, 0.05) !important;
                color: #ffffff !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 8px !important;
                padding: 0.6rem 1.5rem !important;
                font-weight: 500 !important;
                letter-spacing: 0.02em;
                transition: all 0.2s ease !important;
                box-shadow: none !important;
            }

            .stButton > button:hover, .stDownloadButton > button:hover {
                background: rgba(0, 229, 255, 0.1) !important;
                border-color: rgba(0, 229, 255, 0.4) !important;
                color: #00e5ff !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(0, 229, 255, 0.15) !important;
            }

            /* Rank Candidates Button - Vibrant Gradient */
            [data-testid="stMainBlockContainer"] .stButton > button[kind="primary"],
            .stButton > button:has-text("Rank Candidates") {
                background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 100%) !important;
                border: none !important;
                color: #ffffff !important;
                font-weight: 700 !important;
                letter-spacing: 0.05em !important;
                box-shadow: 0 4px 20px rgba(0, 210, 255, 0.3) !important;
            }
            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #00b4d8 0%, #5a1fc9 100%) !important;
                box-shadow: 0 6px 28px rgba(0, 210, 255, 0.45) !important;
                transform: translateY(-3px) !important;
            }

            /* File Uploader */
            .stFileUploader {
                padding: 0.5rem 0;
            }

            div[data-testid="stFileUploader"] section {
                background: rgba(17, 22, 37, 0.6);
                border: 1px dashed rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 1.5rem;
                transition: all 0.2s ease;
            }
            
            div[data-testid="stFileUploader"] section:hover {
                border-color: #00e5ff;
                background: rgba(0, 229, 255, 0.02);
            }

            div[data-testid="stFileUploader"] section div[role="button"] {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #ffffff;
                font-weight: 600;
                border-radius: 10px;
                transition: all 0.2s ease;
            }

            /* DataFrames */
            .stDataFrame, div[data-testid="stDataFrame"] {
                background: rgba(17, 22, 37, 0.6);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                --gdg-bg-cell: transparent;
                --gdg-bg-header: rgba(0, 0, 0, 0.2);
                --gdg-bg-header-has-focus: rgba(0, 229, 255, 0.1);
                --gdg-border-color: rgba(255, 255, 255, 0.05);
                --gdg-text-dark: #ffffff;
                --gdg-text-medium: #cbd5e1;
                --gdg-text-light: #94a3b8;
                --gdg-accent-color: #00e5ff;
            }

            /* Metric Cards Override */
            [data-testid="stMetricValue"] {
                color: #00e5ff !important;
                font-family: 'Outfit', sans-serif;
                font-weight: 700;
            }
            [data-testid="stMetricLabel"] {
                color: #94a3b8 !important;
                font-size: 0.9rem;
            }
            
            /* Expanders */
            [data-testid="stExpander"] {
                background: rgba(17, 22, 37, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 12px !important;
            }
            [data-testid="stExpander"] details summary p {
                color: #ffffff !important;
                font-weight: 600 !important;
            }
            
            /* Highlight */
            .highlight {
                color: #00e5ff;
                font-weight: 600;
            }

            /* Sidebar Expander Buttons */
            [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {
                background: rgba(255, 255, 255, 0.05) !important;
                color: #cbd5e1 !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 6px !important;
                padding: 0.2rem 0.5rem !important;
                font-weight: 500 !important;
                font-size: 0.85rem !important;
                min-height: 2rem !important;
                box-shadow: none !important;
                transition: all 0.2s ease;
            }
            [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button:hover {
                background: rgba(0, 229, 255, 0.1) !important;
                color: #ffffff !important;
                border-color: rgba(0, 229, 255, 0.3) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 8px rgba(0, 229, 255, 0.15) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Intelligent Resume Screening System</div>
            <div class="hero-subtitle">Elevate your hiring with Machine learning powered resume analysis and precise skill matching.</div>
            <div style="margin-top: 1rem;">
                <span class="pill">Skill Context Matrix</span>
                <span class="pill">SVM Engine</span>
                <span class="pill">Actionable Insights</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_recommendations(lacked_skills):
    """Generates improvement recommendations based on lacked skills."""
    recommendations = []
    if not lacked_skills:
        return ["You appear to have all the required skills! Great job!"]

    for skill in lacked_skills:
        recommendations.append(f"To improve in '{skill}', consider taking online courses on platforms like Coursera or Udemy, or work on a project that utilizes this skill.")
    return recommendations

def save_uploaded_file(uploaded_file):
    """Saves an uploaded file temporarily, preserving its extension, and returns the path."""
    if uploaded_file is not None:
        temp_filename = f"temp_{uuid.uuid4()}{os.path.splitext(uploaded_file.name)[1]}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    return None

def calculate_final_score(keyword_score, hard_skill_score, soft_skill_score):
    """Calculates the weighted final score."""
    return (keyword_score * 0.5) + (hard_skill_score * 0.35) + (soft_skill_score * 0.15)


def compute_keyword_scores(processed_jd, processed_resumes):
    """Compute cosine similarity between JD and each resume using a single TF-IDF fit."""
    if not processed_resumes:
        return np.array([], dtype=float)

    if not (processed_jd or "").strip():
        return np.zeros(len(processed_resumes), dtype=float)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([processed_jd] + processed_resumes)
    except ValueError:
        return np.zeros(len(processed_resumes), dtype=float)

    jd_vec = tfidf[0]
    resume_vecs = tfidf[1:]

    # With TF-IDF's default L2 normalization, dot product equals cosine similarity.
    scores = (resume_vecs @ jd_vec.T).toarray().ravel()
    scores = np.clip(scores, 0.0, 1.0)
    return scores

def format_skill_list(skills):
    if not skills:
        return "_None_"
    return "\n".join(f"- {skill}" for skill in skills)

def build_candidate_report(
    candidate_name,
    final_score,
    keyword_score,
    hard_skill_score,
    soft_skill_score,
    svm_prediction,
    svm_confidence,
    matching_skills,
    missing_skills,
):
    lines = [
        f"Candidate: {candidate_name}",
        f"Overall Match Score: {final_score:.2%}",
        "",
        "Score Breakdown",
        f"- Keyword Similarity: {keyword_score:.2%}",
        f"- Hard Skill Match: {hard_skill_score:.2%}",
        f"- Soft Skill Match: {soft_skill_score:.2%}",
        "",
        "SVM Classification",
        f"- Prediction: {svm_prediction}",
    ]

    if svm_confidence is not None:
        lines.append(f"- Confidence: {svm_confidence:.2%}")

    lines.extend([
        "",
        "Matching Skills",
    ])
    lines.extend([f"- {skill}" for skill in matching_skills] or ["- None"])

    lines.extend([
        "",
        "Missing Skills",
    ])
    lines.extend([f"- {skill}" for skill in missing_skills] or ["- None"])

    return "\n".join(lines)

def render_score_wheel(title, keyword_score, hard_skill_score, soft_skill_score, final_score):
    rings = [
        ("Keyword", keyword_score, "#00e5ff", 1.0),
        ("Hard Skills", hard_skill_score, "#a200ff", 0.78),
        ("Soft Skills", soft_skill_score, "#ff00a2", 0.56),
    ]

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    for _, value, color, radius in rings:
        ax.pie(
            [value, max(0.0, 1 - value)],
            radius=radius,
            colors=[color, "#1a2235"],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.18, edgecolor="none"),
        )

    ax.text(0, 0.04, f"{final_score:.0%}", ha="center", va="center", fontsize=12, fontweight="bold", color="#ffffff")
    ax.text(0, -0.16, "Overall", ha="center", va="center", fontsize=8, color="#94a3b8")
    ax.set(aspect="equal")
    ax.axis("off")

    st.caption(title)
    st.pyplot(fig, use_container_width=False)
    st.caption("Keyword (Cyan) • Hard Skills (Purple) • Soft Skills (Pink)")
    plt.close(fig)

def generate_strengths_weaknesses(row):

    strengths = []
    weaknesses = []

    if row["Hard Skill Match"] >= 0.8:
        strengths.append("Strong technical skill alignment")

    if row["Soft Skill Match"] >= 0.8:
        strengths.append("Excellent soft skill compatibility")

    if row["Keyword Match"] >= 0.7:
        strengths.append("Resume closely matches job description")

    if row["Score"] >= 0.8:
        strengths.append("High overall suitability")

    missing = row["Missing Skills"]

    if missing:
        weaknesses.append(
            f"Missing critical skills: {', '.join(missing[:5])}"
        )

    if row["Hard Skill Match"] < 0.5:
        weaknesses.append("Technical skill match is below expectations")

    if row["Soft Skill Match"] < 0.5:
        weaknesses.append("Soft skill alignment could be improved")

    if row["Keyword Match"] < 0.4:
        weaknesses.append(
            "Resume content weakly matches job description"
        )

    return strengths, weaknesses

def render_skill_chips(skills, color):
    if not skills:
        st.caption("None")
        return

    chips_html = ""

    for skill in skills:
        chips_html += f"""
        <span style="
            display:inline-block;
            margin:4px;
            padding:6px 12px;
            border-radius:20px;
            background:{color};
            color:white;
            font-weight:600;
            font-size:14px;
        ">
            {skill}
        </span>
        """

    st.markdown(chips_html, unsafe_allow_html=True)

def _is_extract_error(text):
    return (
        "Unsupported file type" in text
        or "OCR not available" in text
        or "File not found" in text
    )

def _label_from_filename(filename, partial_label):
    name = filename.lower()
    if "good" in name:
        return 1
    if "poor" in name:
        return 0
    if "partial" in name:
        if partial_label == "exclude":
            return None
        return 1 if partial_label == "good" else 0
    return None

def _build_svm_training_data(jd_text, resumes_folder, partial_label):
    processed_jd = preprocess_text(jd_text)
    training_texts = []
    labels = []

    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)
        if os.path.isdir(file_path):
            continue

        label = _label_from_filename(filename, partial_label)
        if label is None:
            continue

        resume_text = extract_text_from_file(file_path)
        if _is_extract_error(resume_text):
            continue

        processed_resume = preprocess_text(resume_text)
        training_texts.append(f"jd {processed_jd} resume {processed_resume}")
        labels.append(label)

    return training_texts, labels

@st.cache_resource
def get_svm_model(jd_path, resumes_folder, partial_label="exclude"):
    if partial_label not in {"exclude", "good", "poor"}:
        raise ValueError("partial_label must be 'exclude', 'good', or 'poor'.")

    jd_text = extract_text_from_file(jd_path)
    if _is_extract_error(jd_text):
        raise ValueError("Training job description file type is not supported.")

    training_texts, labels = _build_svm_training_data(jd_text, resumes_folder, partial_label)
    if not training_texts:
        raise ValueError("No labeled resumes found for SVM training.")

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(training_texts)

    classifier = SVC(kernel='linear', probability=True, class_weight='balanced')
    classifier.fit(X, labels)

    return classifier, vectorizer

def predict_svm_match(classifier, vectorizer, jd_text, resume_text, processed_jd=None, processed_resume=None):
    if processed_jd is None:
        processed_jd = preprocess_text(jd_text)
    if processed_resume is None:
        processed_resume = preprocess_text(resume_text)
    resume_vector = vectorizer.transform([f"jd {processed_jd} resume {processed_resume}"])
    prediction = classifier.predict(resume_vector)[0]
    probability = classifier.predict_proba(resume_vector)[0][prediction]
    return prediction, probability

def applicant_view():
    st.header("🔍 Applicant Panel")
    st.markdown("Get a tailored match score and clear skill feedback in seconds.")

    st.markdown(
        """
        <div class="section-card">
            <span class="highlight">Step 1:</span> Upload a job description and your resume.<br/>
            <span class="highlight">Step 2:</span> Click <b>Analyze</b> to view your match breakdown.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        job_description_file = st.file_uploader("Upload Job Description", type=['txt', 'pdf', 'docx'], key="applicant_jd")
    with col2:
        resume_file = st.file_uploader(
            "Upload Your Resume",
            type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="applicant_resume"
        )

    if st.button("Analyze ✨", use_container_width=True):
        if job_description_file and resume_file:
            with st.spinner("Analyzing... This may take a moment."):
                jd_path, resume_path = None, None
                try:
                    jd_path = save_uploaded_file(job_description_file)
                    resume_path = save_uploaded_file(resume_file)

                    jd_text = extract_text_from_file(jd_path)
                    resume_text = extract_text_from_file(resume_path)

                    if _is_extract_error(jd_text) or _is_extract_error(resume_text):
                        st.error("File type not supported or OCR not available. Use .txt, .pdf, .docx, or image files with OCR installed.")
                        return

                    # --- Skill Extraction ---
                    jd_hard_skills = extract_skills(jd_text, HARD_SKILLS)
                    jd_soft_skills = extract_skills(jd_text, SOFT_SKILLS)
                    resume_hard_skills = extract_skills(resume_text, HARD_SKILLS)
                    resume_soft_skills = extract_skills(resume_text, SOFT_SKILLS)

                    matching_hard_skills = [skill for skill in jd_hard_skills if skill in resume_hard_skills]
                    missing_hard_skills = [skill for skill in jd_hard_skills if skill not in resume_hard_skills]
                    matching_soft_skills = [skill for skill in jd_soft_skills if skill in resume_soft_skills]
                    missing_soft_skills = [skill for skill in jd_soft_skills if skill not in resume_soft_skills]

                    # --- Scoring ---
                    processed_jd = preprocess_text(jd_text)
                    processed_resume = preprocess_text(resume_text)
                    keyword_score = cosine_similarity(vectorize_text([processed_jd, processed_resume])[0])[0, 1]
                    
                    hard_skill_score = len(matching_hard_skills) / len(jd_hard_skills) if jd_hard_skills else 1.0
                    soft_skill_score = len(matching_soft_skills) / len(jd_soft_skills) if jd_soft_skills else 1.0

                    final_score = calculate_final_score(keyword_score, hard_skill_score, soft_skill_score)

                    svm_prediction = None
                    svm_confidence = None
                    try:
                        svm_classifier, svm_vectorizer = get_svm_model(
                            TRAINING_JD_PATH,
                            TRAINING_RESUMES_FOLDER,
                            partial_label=PARTIAL_LABEL
                        )
                        svm_prediction, svm_confidence = predict_svm_match(
                            svm_classifier,
                            svm_vectorizer,
                            jd_text,
                            resume_text,
                            processed_jd=processed_jd,
                            processed_resume=processed_resume,
                        )
                    except ValueError as exc:
                        st.warning(str(exc))

                    with st.container(border=True):

                        left_col, right_col = st.columns([2, 1])

                        with left_col:

                            st.markdown("## 🎯 Applicant Report")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Score", f"{final_score:.0%}")

                            with col2:
                                st.metric(
                                    "Matched",
                                    len(matching_hard_skills) + len(matching_soft_skills)
                                )

                            with col3:
                                st.metric(
                                    "Missing",
                                    len(missing_hard_skills) + len(missing_soft_skills)
                                )

                            strengths = []

                            if soft_skill_score >= 0.7:
                                strengths.append("Excellent soft skill compatibility")

                            if hard_skill_score >= 0.7:
                                strengths.append("Strong technical skill alignment")

                            if keyword_score >= 0.6:
                                strengths.append("Resume content aligns well with job description")

                            weaknesses = []

                            if missing_hard_skills:
                                weaknesses.append(
                                    f"Missing critical skills: {', '.join(missing_hard_skills[:5])}"
                                )

                            if keyword_score < 0.5:
                                weaknesses.append(
                                    "Resume content weakly matches job description"
                                )

                            st.markdown("---")

                            st.markdown("### Strengths")

                            for s in strengths:
                                st.success(s)

                            st.markdown("### Weaknesses")

                            for w in weaknesses:
                                st.warning(w)

                            st.markdown("---")

                            st.markdown("### Matching Skills")

                            all_matching = matching_hard_skills + matching_soft_skills

                            st.write(
                                ", ".join(
                                    skill.title()
                                    for skill in all_matching[:10]
                                )
                            )

                            st.markdown("### Missing Skills")

                            all_missing = missing_hard_skills + missing_soft_skills

                            st.write(
                                ", ".join(
                                    skill.title()
                                    for skill in all_missing[:10]
                                )
                            )

                        with right_col:

                            render_score_wheel(
                                "Score Breakdown",
                                keyword_score,
                                hard_skill_score,
                                soft_skill_score,
                                final_score,
                            )

                    st.subheader("SVM Classification")
                    if svm_prediction is None:
                        st.write("Prediction: Unavailable (missing training data)")
                    else:
                        svm_label = "Good Match" if svm_prediction == 1 else "Poor Match"
                        st.write(f"Prediction: **{svm_label}** (Confidence: {svm_confidence:.2%})")

                    st.subheader("🚀 Recommended Learning Path")

                    recommendations = get_recommendations(
                        missing_hard_skills + missing_soft_skills
                    )

                    for rec in recommendations[:5]:
                        st.info(f"📘 {rec}")
                finally:
                    if jd_path and os.path.exists(jd_path): os.remove(jd_path)
                    if resume_path and os.path.exists(resume_path): os.remove(resume_path)
        else:
            st.error("Please upload both a job description and a resume.")


def process_single_resume(args):
    resume_path, resume_filename, jd_text, processed_jd, jd_hard_skills, jd_soft_skills, classifier, vectorizer = args
    resume_text = extract_text_from_file(resume_path)
    if _is_extract_error(resume_text):
        return None, f"Skipping {resume_filename}: {resume_text}"

    processed_resume = preprocess_text(resume_text)
    tfidf_matrix, _ = vectorize_text([processed_jd, processed_resume])
    keyword_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]

    resume_hard_skills = extract_skills(resume_text, HARD_SKILLS)
    resume_soft_skills = extract_skills(resume_text, SOFT_SKILLS)

    matching_hard_skills = [s for s in jd_hard_skills if s in resume_hard_skills]
    matching_soft_skills = [s for s in jd_soft_skills if s in resume_soft_skills]
    missing_hard_skills = [s for s in jd_hard_skills if s not in resume_hard_skills]
    missing_soft_skills = [s for s in jd_soft_skills if s not in resume_soft_skills]

    hard_skill_score = len(matching_hard_skills) / len(jd_hard_skills) if jd_hard_skills else 1.0
    soft_skill_score = len(matching_soft_skills) / len(jd_soft_skills) if jd_soft_skills else 1.0
    final_score = calculate_final_score(keyword_score, hard_skill_score, soft_skill_score)

    svm_prediction = "Unavailable"
    svm_confidence = None
    if classifier is not None:
        svm_prediction_value, svm_confidence = predict_svm_match(
            classifier,
            vectorizer,
            jd_text,
            resume_text,
            processed_jd=processed_jd,
            processed_resume=processed_resume,
        )
        svm_prediction = "Good Match" if svm_prediction_value == 1 else "Poor Match"

    report_text = build_candidate_report(
        resume_filename, final_score, keyword_score, hard_skill_score, soft_skill_score,
        svm_prediction, svm_confidence, matching_hard_skills + matching_soft_skills,
        missing_hard_skills + missing_soft_skills,
    )

    result = {
        "File Name": resume_filename,
        "Score": final_score,
        "Keyword Match": keyword_score,
        "Hard Skill Match": hard_skill_score,
        "Soft Skill Match": soft_skill_score,
        "SVM Prediction": svm_prediction,
        "SVM Confidence": svm_confidence,
        "Matching Skills": matching_hard_skills + matching_soft_skills,
        "Missing Skills": missing_hard_skills + missing_soft_skills,
        "Report Name": f"{os.path.splitext(resume_filename)[0]}_report.txt",
        "Report Bytes": report_text.encode("utf-8"),
    }
    return result, None

def process_resume_batch(batch_args):
    """Processes a batch of resumes sequentially within a single worker."""
    batch_results = []
    batch_errors = []
    for args in batch_args:
        try:
            res, err = process_single_resume(args)
            batch_results.append(res)
            batch_errors.append(err)
        except Exception as e:
            batch_results.append(None)
            batch_errors.append(str(e))
    return batch_results, batch_errors

def manager_view():
    st.header("👔 Manager Panel")
    st.markdown("Upload a job description and multiple resumes to rank candidates quickly.")

    st.markdown(
        """
        <div class="section-card">
            <span class="highlight">Tip:</span> For faster results, upload resumes as <b>.txt</b>, <b>.pdf</b>, or <b>.docx</b> files.
        </div>
        """,
        unsafe_allow_html=True,
    )

    job_description_file = st.file_uploader("Upload Job Description", type=['txt', 'pdf', 'docx'], key="manager_jd")
    resume_files = st.file_uploader(
        "Upload Resumes",
        type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        key="manager_resumes"
    )

    if st.button("Rank Candidates 🚀", use_container_width=True, type="primary"):
        if job_description_file and resume_files:
            progress = st.progress(0)
            status = st.empty()
            try:
                jd_text = extract_text_from_bytes(job_description_file.getvalue(), job_description_file.name)
                if _is_extract_error(jd_text):
                    st.error("Job description file type not supported or OCR not available.")
                    return
                # Manager bulk ranking uses a fast TF-IDF pipeline directly on raw text.
                jd_for_similarity = jd_text
                jd_hard_skills = extract_skills(jd_text, HARD_SKILLS)
                jd_soft_skills = extract_skills(jd_text, SOFT_SKILLS)

                svm_classifier = None
                svm_vectorizer = None
                try:
                    svm_classifier, svm_vectorizer = get_svm_model(
                        TRAINING_JD_PATH,
                        TRAINING_RESUMES_FOLDER,
                        partial_label=PARTIAL_LABEL
                    )
                except ValueError as exc:
                    st.warning(str(exc))

                rank_start = time.perf_counter()

                resume_items = [(rf.name, rf.getvalue()) for rf in resume_files]

                total = len(resume_items)
                if total == 0:
                    st.error("No resumes were saved successfully. Please try uploading again.")
                    return

                status.text("Extracting resume text...")
                extracted_texts = [None] * total
                extracted_errors = [None] * total

                max_workers = min(32, (os.cpu_count() or 4) * 2)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(extract_text_from_bytes, resume_bytes, resume_filename): i
                        for i, (resume_filename, resume_bytes) in enumerate(resume_items)
                    }

                    extracted_count = 0
                    for future in concurrent.futures.as_completed(future_to_index):
                        i = future_to_index[future]
                        try:
                            resume_text = future.result()
                        except Exception as exc:
                            resume_text = f"Error extracting text: {exc}"

                        if _is_extract_error(resume_text) or not (resume_text or "").strip():
                            extracted_errors[i] = resume_text or "Empty text extracted."
                        else:
                            extracted_texts[i] = resume_text

                        extracted_count += 1
                        progress.progress(min((extracted_count / total) * 0.6, 0.6))
                        status.text(f"Extracted {extracted_count}/{total}...")

                valid_records = []
                for i, (resume_filename, _) in enumerate(resume_items):
                    resume_text = extracted_texts[i]
                    if resume_text is None:
                        st.warning(f"Skipping {resume_filename}: {extracted_errors[i]}")
                        continue
                    valid_records.append({
                        "File Name": resume_filename,
                        "Text": resume_text,
                    })

                if not valid_records:
                    st.error("No valid resumes found to rank. Please upload supported files.")
                    return

                status.text("Computing keyword similarity...")
                resume_texts = [r["Text"] for r in valid_records]
                keyword_scores = compute_keyword_scores(jd_for_similarity, resume_texts)

                results = []
                valid_total = len(valid_records)
                status.text("Scoring and building reports...")

                for idx, record in enumerate(valid_records):
                    resume_filename = record["File Name"]
                    resume_text = record["Text"]
                    keyword_score = float(keyword_scores[idx]) if idx < len(keyword_scores) else 0.0

                    resume_hard_skills = extract_skills(resume_text, HARD_SKILLS)
                    resume_soft_skills = extract_skills(resume_text, SOFT_SKILLS)

                    matching_hard_skills = [s for s in jd_hard_skills if s in resume_hard_skills]
                    matching_soft_skills = [s for s in jd_soft_skills if s in resume_soft_skills]
                    missing_hard_skills = [s for s in jd_hard_skills if s not in resume_hard_skills]
                    missing_soft_skills = [s for s in jd_soft_skills if s not in resume_soft_skills]

                    hard_skill_score = len(matching_hard_skills) / len(jd_hard_skills) if jd_hard_skills else 1.0
                    soft_skill_score = len(matching_soft_skills) / len(jd_soft_skills) if jd_soft_skills else 1.0
                    final_score = calculate_final_score(keyword_score, hard_skill_score, soft_skill_score)

                    svm_prediction = "Unavailable"
                    svm_confidence = None
                    if svm_classifier is not None:
                        svm_prediction_value, svm_confidence = predict_svm_match(
                            svm_classifier,
                            svm_vectorizer,
                            jd_text,
                            resume_text,
                            # Pass raw text so we don't pay spaCy cost in bulk.
                            processed_jd=jd_text,
                            processed_resume=resume_text,
                        )
                        svm_prediction = "Good Match" if svm_prediction_value == 1 else "Poor Match"

                    report_text = build_candidate_report(
                        resume_filename,
                        final_score,
                        keyword_score,
                        hard_skill_score,
                        soft_skill_score,
                        svm_prediction,
                        svm_confidence,
                        matching_hard_skills + matching_soft_skills,
                        missing_hard_skills + missing_soft_skills,
                    )

                    result = {
                        "File Name": resume_filename,
                        "Score": final_score,
                        "Keyword Match": keyword_score,
                        "Hard Skill Match": hard_skill_score,
                        "Soft Skill Match": soft_skill_score,
                        "SVM Prediction": svm_prediction,
                        "SVM Confidence": svm_confidence,
                        "Matching Skills": matching_hard_skills + matching_soft_skills,
                        "Missing Skills": missing_hard_skills + missing_soft_skills,
                        "Report Name": f"{os.path.splitext(resume_filename)[0]}_report.txt",
                        "Report Bytes": report_text.encode("utf-8"),
                    }
                    results.append(result)

                    progress.progress(0.6 + ((idx + 1) / valid_total) * 0.4)
                    status.text(f"Processed {idx + 1}/{valid_total}...")

                status.text("Ranking complete.")
                st.caption(f"Processed {valid_total} resumes in {time.perf_counter() - rank_start:.1f}s")

                if not results:
                    st.error("No valid resumes found to rank. Please upload supported files.")
                    return

                df = pd.DataFrame(results)
                df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
                st.session_state.ranked_df = df
                st.session_state.ranking_done = True
                
                # --- Store in History ---
                history_id = str(uuid.uuid4())
                st.session_state.history.insert(0, {
                    "id": history_id,
                    "timestamp": datetime.now(),
                    "job_description": job_description_file.name,
                    "results_df": df,
                })
                st.session_state.selected_history_id = history_id
                # ----------------------

                
            finally:
                pass
        else:
            st.error("Please upload a job description and at least one resume.")
    if st.session_state.ranking_done:

        df = st.session_state.ranked_df

        svm_confidence_format = "{:.2%}"

        st.subheader("Ranked Candidates")

        st.dataframe(
            df[['File Name', 'Score', 'Keyword Match',
                'Hard Skill Match', 'Soft Skill Match',
                'SVM Prediction', 'SVM Confidence']].style.format({
                "Score": "{:.2%}",
                "Keyword Match": "{:.2%}",
                "Hard Skill Match": "{:.2%}",
                "Soft Skill Match": "{:.2%}",
                "SVM Confidence": svm_confidence_format
            }),
            use_container_width=True
        )

        st.subheader("Detailed View")

        selected_candidate = st.selectbox(
            "Select Candidate",
            df["File Name"].tolist(),
            key="candidate_selector"
        )

        row = df[df["File Name"] == selected_candidate].iloc[0]
        with st.container(border=True):

            left_col, right_col = st.columns([2,1])

            with left_col:
                st.markdown("## 👤 Candidate Profile")

                st.write(f"**Resume:** {selected_candidate}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Score",
                        f"{row['Score']:.0%}"
                    )

                with col2:
                    st.metric(
                        "Matched",
                        len(row["Matching Skills"])
                    )

                with col3:
                    st.metric(
                        "Missing",
                        len(row["Missing Skills"])
                    )
                strengths, weaknesses = generate_strengths_weaknesses(row)

                st.markdown("---")

                st.markdown("#### Strengths")

                for s in strengths:
                    st.markdown(f"🔸 {s}")

                st.markdown("#### Weaknesses")

                for w in weaknesses:
                    st.markdown(f"🔸 {w}")

                st.markdown("#### Matching Skills")

                st.write(
                    ", ".join(
                        skill.title()
                        for skill in row["Matching Skills"][:10]
                    )
                )

                st.markdown("#### Missing Skills")

                st.write(
                    ", ".join(
                        skill.title()
                        for skill in row["Missing Skills"][:10]
                    )
                )
            with right_col:

                render_score_wheel(
                    "Score Breakdown",
                    row["Keyword Match"],
                    row["Hard Skill Match"],
                    row["Soft Skill Match"],
                    row["Score"]
                )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="download-btn">', unsafe_allow_html=True)

            st.download_button(
                "📄 Download Analysis Report",
                data=row["Report Bytes"],
                file_name=row["Report Name"],
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="clear-btn">', unsafe_allow_html=True)

                clear = st.button(
                    "🗑 Clear Results",
                    use_container_width=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

        if clear:
            st.session_state.ranked_df = None
            st.session_state.ranking_done = False
            st.rerun()

def main():
    st.set_page_config(page_title="Intelligent Resume Screener", page_icon="📄", layout="wide")
    
    if "history" not in st.session_state:
        st.session_state.history = []

    if "ranked_df" not in st.session_state:
        st.session_state.ranked_df = None

    if "ranking_done" not in st.session_state:
        st.session_state.ranking_done = False

    inject_custom_css()
    render_hero()
    
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Go to", ["Applicant View", "Manager View"])
    
    if view == "Applicant View":
        applicant_view()
    else:
        if render_manager_auth():
            manager_view()

    st.sidebar.markdown("---")
    st.sidebar.info("This app helps you match resumes to job descriptions and rank candidates.")

    if view == "Manager View" and st.session_state.get("manager_logged_in") and st.session_state.history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Analysis History")
        for i, entry in enumerate(st.session_state.history):
            with st.sidebar.expander(
                f"{entry['timestamp'].strftime('%b %d, %H:%M')} - {entry['job_description']}"
            ):
                view_clicked = False
                delete_clicked = False

                st.write(f"{len(entry['results_df'])} candidates")

                col1, col2 = st.columns(2)

                with col1:
                    view_clicked = st.button(
                        "👁 View",
                        key=f"view_{entry['id']}",
                        use_container_width=True
                    )

                with col2:
                    delete_clicked = st.button(
                        "🗑 Delete",
                        key=f"delete_{entry['id']}",
                        use_container_width=True
                    )

                if view_clicked:
                    st.session_state.selected_history_id = entry['id']
                    st.session_state.ranked_df = entry['results_df']
                    st.session_state.ranking_done = True
                    st.rerun()
                if delete_clicked:
                    st.session_state.history.pop(i)
                    if st.session_state.get('selected_history_id') == entry['id']:
                        st.session_state.selected_history_id = None
                    st.rerun()


if __name__ == "__main__":
    main()
