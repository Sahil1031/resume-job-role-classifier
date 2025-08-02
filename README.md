# 📄 Resume Job Role Classifier

A machine learning and NLP-based web application that predicts the most suitable job category based on the content of a resume. Built using Python, Scikit-learn, and Streamlit.

---

## 🚀 Features

- Upload resume files in `.pdf` or `.docx` format
- Automatically extract and clean resume text
- Predict job role from pre-defined categories using a trained ML model
- Clean and interactive UI built with Streamlit
- Ready for deployment on Streamlit Cloud

---

## 💡 Use Case

This app simulates an automated resume screening system — helping recruiters or applicants match resumes with relevant job categories like:

- Data Scientist  
- Java Developer  
- DevOps Engineer  
- HR  
- Python Developer  
- ...and more

---

## 🧠 Technologies Used

- **Python 3.8+**
- **Scikit-learn** (Logistic Regression, LabelEncoder)
- **TfidfVectorizer** for text vectorization
- **Streamlit** for frontend deployment
- **PyMuPDF** (`fitz`) for PDF parsing
- **python-docx** for DOCX extraction
- **joblib** for saving/loading models
