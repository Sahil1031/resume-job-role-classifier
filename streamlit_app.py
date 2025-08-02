import streamlit as st
import joblib
import fitz
import docx
import re

model = joblib.load("resume_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|mailto:\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

st.set_page_config(page_title="Resume Classifier", layout="centered")

st.markdown("<h1 style='text-align: center;'>📄 Resume Job Role Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload your resume to find out the most suitable job category using Machine Learning and NLP.</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload your resume (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == 'docx':
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        resume_text = ""

    if resume_text:
        st.subheader("📄 Resume Text Preview")
        st.text_area(label="", value=resume_text[:2000], height=250)

        cleaned = clean_text(resume_text)
        vector = tfidf.transform([cleaned]).toarray()
        pred = model.predict(vector)
        predicted_label = le.inverse_transform(pred)[0]

        st.markdown("---")
        st.markdown("### 🎯 Predicted Job Category:")
        st.success(f"{predicted_label}")
        st.balloons()
    else:
        st.warning("⚠️ Could not extract text from file.")
else:
    st.info("📂 Please upload a PDF or DOCX resume to get started.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Made with ❤️ by <b>Sahil Bagde</b></p>",
    unsafe_allow_html=True
)
