# Smart Resume Analyzer - Beginner Friendly Web App
# Run with: streamlit run app.py

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Helper functions
# -------------------------------

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()


def extract_skills(text):
    skills_list = [
        "python", "java", "c", "c++", "sql", "html", "css",
        "machine learning", "data science", "django", "flask",
        "excel", "power bi", "communication", "teamwork"
    ]
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found


def calculate_match(resume_text, jd_text):
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3) 
          )# allows phrases like machine learning
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors)[0][1]
    return round(score * 100, 2)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")

st.title("🧠 Smart Resume Analyzer")
st.write("Upload your resume and compare it with a job description.")

uploaded_resume = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description here")

if st.button("🔍 Analyze Resume"):
    if uploaded_resume is None or job_description.strip() == "":
        st.error("Please upload a resume and enter a job description")
    else:
        resume_text = extract_text_from_pdf(uploaded_resume)
        jd_text = job_description.lower()

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        match_score = calculate_match(resume_text, jd_text)
        missing_skills = set(jd_skills) - set(resume_skills)

        st.subheader("📊 Analysis Result")
        st.metric("Resume Match Score", f"{match_score}%")

        st.subheader("✅ Skills Found in Resume")
        if resume_skills:
            st.write(resume_skills)
        else:
            st.write("No listed skills found")

        st.subheader("❌ Missing Skills")
        if missing_skills:
            st.write(list(missing_skills))
        else:
            st.write("No missing skills 🎉")

        st.subheader("💡 Suggestions")
        if match_score < 60:
            st.warning("Improve skills and add relevant projects")
            st.warning("Weak match score")
        elif match_score < 80:
            st.info("Add missing skills and improve resume wording")
            st.warning("Medium match score")
        else:
            st.success("Your resume matches well with the job description")
            st.warning("Strong match score")
st.markdown("---")
st.caption("Internship Project | Smart Resume Analyzer")