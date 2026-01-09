# 🧠 AI Text Intelligence Dashboard (NLTK)

A production-ready **Natural Language Processing (NLP) web application** that analyzes raw text to extract sentiment, keywords, and named entities — complete with confidence scoring, keyword highlighting, downloadable reports, and live deployment.

🚀 **Live Demo:**  
https://sihabsafin-nltk-text-intelligence.streamlit.app/

---

## 📌 Project Overview

The **AI Text Intelligence Dashboard** is an end-to-end NLP application built using **Python, NLTK, and Streamlit**.  
It demonstrates how classical NLP techniques can be combined into a modern, interactive, and deployable web product.

This project focuses on:
- Practical NLP pipelines
- Explainable AI outputs
- Clean UI/UX
- Real-world deployment on Streamlit Cloud

---

## ✨ Key Features

### 🔍 Text Analysis
- Sentence & document-level text processing
- Tokenization, stopword removal, and lemmatization

### 📊 Sentiment Analysis
- Binary sentiment classification (Positive / Negative)
- **Confidence score** using probabilistic classification

### 🔑 Keyword Extraction
- Automatic keyword identification
- **Highlighted keywords inside the original text** for better interpretability

### 🏷️ Named Entity Recognition (NER)
- Detects real-world entities such as:
  - Persons
  - Organizations
  - Locations
- Powered by NLTK’s MaxEnt NER chunker

### 📄 Report Export
- **Download analysis results as JSON**
- **Download professional PDF reports** (HR & stakeholder friendly)

### 🕘 Analysis History
- Session-based history of previous text analyses
- Enables quick comparison of results

### ☁️ Live Deployment
- Fully deployed on **Streamlit Cloud**
- Stable with **Python 3.13** and latest NLTK resources

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **NLP Library:** NLTK  
- **Web Framework:** Streamlit  
- **Report Generation:** ReportLab  
- **Deployment:** Streamlit Cloud  

---

## 🧠 NLP Techniques Used

- Tokenization  
- Stopword Removal  
- Lemmatization  
- Feature Engineering (Bag-of-Words style)  
- Naive Bayes Classification  
- Probabilistic Confidence Scoring  
- Part-of-Speech Tagging  
- Named Entity Recognition  

---
## 📈 Why This Project Matters

This project demonstrates:

- Practical understanding of **NLP fundamentals**
- Ability to **build, debug, and deploy** real-world applications
- Experience handling **production issues** (NLTK resource changes, cloud compatibility)
- Clean and professional **UI/UX design thinking**
- Clear and **explainable AI outputs**

It goes beyond tutorials and reflects **real-world machine learning engineering skills**.

---

## 🔮 Future Enhancements

- spaCy-based NLP pipeline
- Transformer-based sentiment analysis (HuggingFace)
- Multi-language support
- Database-backed history (Supabase / Firebase)
- User authentication and user-specific analytics

---

## 👨‍💻 Author

**Sihab Safin**  
Machine Learning & NLP Enthusiast  

🔗 **GitHub:** https://github.com/sihabsafin/nltk-text-intelligence/
🌐 **Live App:** https://sihabsafin-nltk-text-intelligence.streamlit.app/

## 🚀 How to Run Locally

> ### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/sihabsafin/nltk-text-intelligence.git
cd nltk-text-intelligence

---------------------
> ###  🔹 Step 2: Install Dependencies

pip install -r requirements.txt
---------------------
> ###  🔹 Step 3: Run the Application

streamlit run app.py
---------------------
> ###  🔹 Step 4: 📂 Project Structure

nltk-text-intelligence/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation


---







