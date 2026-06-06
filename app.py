import streamlit as st
import pandas as pd
import os
import time
import tempfile
import concurrent.futures
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

def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

            .stApp {
                background: linear-gradient(135deg, #f8fbff 0%, #f1f7ff 45%, #f4fff8 100%);
            }

            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
                color: #0f172a;
            }

            .stApp p, .stApp li, .stApp span, .stApp label {
                color: #1f2937;
            }

            .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div, .stMarkdown li {
                color: #1f2937;
            }

            [data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid #e5e7eb;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div {
                color: #0f172a;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }

            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.01em;
            }

            .stApp {
                font-family: 'IBM Plex Sans', sans-serif;
            }

            .hero {
                padding: 1.5rem 1.75rem;
                border-radius: 18px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
                margin-bottom: 1.25rem;
            }

            .hero-title {
                font-size: 2.1rem;
                margin-bottom: 0.25rem;
                color: #0f172a;
            }

            .hero-subtitle {
                color: #475569;
                font-size: 1rem;
                margin-bottom: 0.75rem;
            }

            .pill {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                font-size: 0.85rem;
                background: #e6f6ff;
                color: #0b5cab;
                margin-right: 0.35rem;
            }

            .section-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
                color: #0f172a;
            }

            .stButton > button {
                background: linear-gradient(90deg, #38bdf8, #22c55e);
                color: #ffffff;
                border: none;
                border-radius: 12px;
                padding: 0.6rem 1.2rem;
                font-weight: 600;
                letter-spacing: 0.01em;
                transition: transform 0.1s ease, box-shadow 0.1s ease;
            }

            .stDownloadButton > button {
                background: #22c55e !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 18px rgba(56, 189, 248, 0.25);
            }

            .stFileUploader {
                padding: 0.4rem 0;
            }

            div[data-testid="stFileUploader"] section {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 12px;
                padding: 0.75rem;
            }

            div[data-testid="stFileUploader"] section div[role="button"] {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                color: #0f172a;
                font-weight: 600;
                border-radius: 10px;
            }

            div[data-testid="stFileUploader"] section button,
            div[data-testid="stFileUploader"] section button * {
                background: #ffffff !important;
                color: #0f172a !important;
                border-color: #cbd5e1 !important;
            }

            div[data-testid="stFileUploader"] section button:hover,
            div[data-testid="stFileUploader"] section button:hover * {
                border-color: #cbd5e1 !important;
                box-shadow: none !important;
                transform: none;
            }

            div[data-testid="stFileUploader"] section button:focus,
            div[data-testid="stFileUploader"] section button:focus-visible {
                outline: none !important;
                box-shadow: none !important;
            }

            div[data-testid="stFileUploader"] section button:active,
            div[data-testid="stFileUploader"] section button:active * {
                transform: translateY(1px) scale(0.98);
                box-shadow: inset 0 2px 6px rgba(15, 23, 42, 0.12) !important;
            }

            div[data-testid="stFileUploader"] section button:disabled,
            div[data-testid="stFileUploader"] section button:disabled * {
                background: #f8fafc !important;
                color: #94a3b8 !important;
                border-color: #e2e8f0 !important;
            }

            div[data-testid="stFileUploader"] section div[role="button"]:hover {
                border-color: #38bdf8;
                box-shadow: 0 6px 12px rgba(56, 189, 248, 0.18);
            }

            div[data-testid="stFileUploader"] small {
                color: #64748b;
            }

            div[data-testid="stFileUploader"] label {
                color: #0f172a;
            }

            .stDataFrame,
            div[data-testid="stDataFrame"] {
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                --gdg-bg-cell: #ffffff;
                --gdg-bg-header: #f8fafc;
                --gdg-bg-header-has-focus: #eef2ff;
                --gdg-border-color: #e2e8f0;
                --gdg-text-dark: #0f172a;
                --gdg-text-medium: #475569;
                --gdg-text-light: #94a3b8;
                --gdg-accent-color: #22c55e;
                --gdg-accent-fg: #ffffff;
                --gdg-accent-light: #bbf7d0;
                --gdg-link-color: #0ea5e9;
                --gdg-rounding-radius: 8px;
            }

            div[data-testid="stDataFrame"] {
                overflow: visible !important;
            }

            div[data-testid="stDataFrame"] button {
                opacity: 1 !important;
                visibility: visible !important;
            }

            div[data-testid="stDataFrame"] svg {
                opacity: 1 !important;
                visibility: visible !important;
            }

            .highlight {
                color: #0f172a;
                font-weight: 600;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] > div[role="button"] p,
            [data-testid="stSidebar"] [data-testid="stExpander"] > div[role="button"] p,
            [data-testid="stSidebar"] [data-testid="stExpander"] button[role="button"] p,
            [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderToggle"] p {
                color: #ffffff;
                font-weight: 600;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] p {
                color: #0f172a;
            }

            [data-testid="stSidebar"] [data-testid="stPopoverButton"] button {
                width: 28px !important;
                height: 28px !important;
                min-height: 28px !important;

                border-radius: 6px !important;

                background: #f1f5f9 !important;
                color: #475569 !important;

                border: 1px solid #e2e8f0 !important;

                padding: 0 !important;

                display: flex !important;
                align-items: center !important;
                justify-content: center !important;

                box-shadow: none !important;
            }

            [data-testid="stSidebar"] [data-testid="stPopoverButton"] button:hover {
                background: #cbd5e1 !important;
            }

            [data-testid="stPopoverBody"] {
                background: white !important;

                border: 1px solid #e5e7eb !important;

                border-radius: 10px !important;

                padding: 6px !important;

                min-width: 120px !important;

                box-shadow: 0 8px 20px rgba(0,0,0,0.12) !important;
            }

            div[data-testid="stPopoverBody"] .stButton > button {

                width: 100% !important;

                background: white !important;

                color: #0f172a !important;

                border: none !important;

                border-radius: 6px !important;

                text-align: left !important;

                padding: 8px 10px !important;

                font-size: 14px !important;

                box-shadow: none !important;
            }

            div[data-testid="stPopoverBody"] .stButton > button:hover {
                background: #f8fafc !important;
                color: #0f172a !important;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] .stButton button {
                background: #f1f5f9 !important;
                color: #475569 !important;
                border-radius: 6px !important;
                padding: 0.05rem 0.45rem !important;
                font-weight: 600;
                font-size: 0.72rem;
                min-height: 1.4rem;
                line-height: 1.1;
                border: 1px solid #e2e8f0 !important;
                box-shadow: none !important;
            }
            [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            margin-bottom: 10px !important;
            overflow: hidden !important;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] > div[role="button"] {
                background: #ffffff !important;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] > div[role="button"] p {
                color: #0f172a !important;
                font-weight: 600 !important;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] .stButton button:hover {
                background: #e2e8f0 !important;
                color: #1e293b !important;
                transform: none;
                box-shadow: none;
            }
                background: #e2e8f0;
                color: #1e293b;
                transform: none;
                box-shadow: none;
            }

            [data-testid="stExpander"] details summary {
                background: #ffffff !important;
                color: #0f172a !important;
                border-radius: 10px !important;
            }

            [data-testid="stExpander"] details summary p {
                color: #0f172a !important;
                font-weight: 600 !important;
            }

            [data-testid="stExpander"] details {
                background: #ffffff !important;
            }

            [data-testid="stExpander"] {
                background: #ffffff !important;
            }

            div[data-testid="stDataFrame"] button {
                background: transparent !important;
                color: #64748b !important;
                border: none !important;
                box-shadow: none !important;
            }

            div[data-testid="stDataFrame"] button:hover {
                background: #f1f5f9 !important;
                color: #0f172a !important;
            }

            div[data-testid="stDataFrame"] svg {
                fill: #64748b !important;
                color: #64748b !important;
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
            <div class="hero-subtitle">Rank candidates faster with skill-aware matching, SVM insights, and clarity-first scoring.</div>
            <span class="pill">Skill Match</span>
            <span class="pill">SVM Classification</span>
            <span class="pill">Readable Insights</span>
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
        ("Keyword", keyword_score, "#38bdf8", 1.0),
        ("Hard Skills", hard_skill_score, "#22c55e", 0.78),
        ("Soft Skills", soft_skill_score, "#f59e0b", 0.56),
    ]

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    for _, value, color, radius in rings:
        ax.pie(
            [value, max(0.0, 1 - value)],
            radius=radius,
            colors=[color, "#e5e7eb"],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.18, edgecolor="white"),
        )

    ax.text(0, 0.04, f"{final_score:.0%}", ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")
    ax.text(0, -0.16, "Overall", ha="center", va="center", fontsize=8, color="#64748b")
    ax.set(aspect="equal")
    ax.axis("off")

    st.caption(title)
    st.pyplot(fig, use_container_width=False)
    st.caption("Keyword (blue) • Hard Skills (green) • Soft Skills (orange)")
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

    if st.button("Rank Candidates 🚀", use_container_width=True):
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
        manager_view()

    st.sidebar.markdown("---")
    st.sidebar.info("This app helps you match resumes to job descriptions and rank candidates.")

    if view == "Manager View" and st.session_state.history:
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
                if delete_clicked:
                    st.session_state.history.pop(i)
                    if st.session_state.get('selected_history_id') == entry['id']:
                        st.session_state.selected_history_id = None
                    st.rerun()


if __name__ == "__main__":
    main()
