# 📰 TruthLens – AI-Powered Multilingual Fake News Detection & News Credibility Analysis

## 📌 Project Overview

TruthLens is an AI-powered web application designed to combat misinformation by automatically analyzing news articles and predicting their credibility. The platform leverages state-of-the-art Natural Language Processing (NLP) and Machine Learning techniques to classify news as **Real** or **Fake**, while providing users with confidence scores and explainable predictions.

Unlike traditional fake news detection systems that support only English, TruthLens is designed as a **multilingual platform**, enabling users to verify news articles from different languages through automatic translation and AI-based credibility analysis.

The project combines modern transformer-based language models, explainable AI, and an interactive web interface to provide an accessible fact-checking solution for journalists, researchers, students, and the general public.

---

# 🎯 Problem Statement

The rapid spread of misinformation through online news platforms and social media has made it increasingly difficult to distinguish between authentic and fake news. Manual fact-checking is time-consuming and cannot keep pace with the volume of digital content.

This project aims to develop an intelligent AI system capable of automatically detecting misleading news articles and assisting users in verifying the credibility of online information.

---

# 🚀 Objectives

- Detect fake and misleading news articles using NLP.
- Support multilingual news verification.
- Provide confidence scores for predictions.
- Improve transparency through explainable AI techniques.
- Build an interactive web application for real-time news analysis.
- Assist users in making informed decisions based on credible information.

---

# ✨ Key Features

- AI-powered Fake News Detection
- Multilingual News Analysis
- Automatic Language Translation
- Confidence Score Prediction
- Explainable AI-based Predictions
- Clean and Interactive Web Interface
- Real-time Article Classification
- Transformer-based NLP Models
- Fast Inference Pipeline

---

# 🧠 Project Architecture

```
User Input
      │
      ▼
News Article / Headline
      │
      ▼
Language Detection
      │
      ▼
Translation (if required)
      │
      ▼
Text Preprocessing
      │
      ▼
Transformer-based NLP Model
      │
      ▼
Prediction
      │
      ▼
Credibility Score
      │
      ▼
Real / Fake Classification
      │
      ▼
Interactive Dashboard
```

---

# ⚙️ Project Workflow

## Step 1 – User Input

The user enters:

- News headline
- News article
- Online news content

---

## Step 2 – Language Processing

The application automatically:

- Detects the input language
- Translates non-English text into English (if required)
- Normalizes the text

---

## Step 3 – Text Preprocessing

The input text undergoes:

- Lowercasing
- Removal of unnecessary characters
- Tokenization
- Text normalization

before being passed to the prediction model.

---

## Step 4 – AI Prediction

The processed text is analyzed using a transformer-based NLP model trained for fake news classification.

The model predicts:

- Real News
- Fake News

along with prediction confidence.

---

## Step 5 – Explainable Results

The system displays:

- Prediction
- Confidence Score
- Probability Distribution
- User-friendly output for better understanding.

---

# 📂 Dataset

The model is trained using publicly available fake news datasets consisting of:

- Real news articles
- Fake news articles
- Headlines
- Full article content
- Multiple news categories

The dataset was preprocessed before training to improve model generalization and prediction accuracy.

---

# 🧠 Machine Learning Pipeline

The project follows the complete NLP workflow:

```
Dataset
    │
    ▼
Cleaning
    │
    ▼
Tokenization
    │
    ▼
Transformer Tokenizer
    │
    ▼
Model Training
    │
    ▼
Validation
    │
    ▼
Testing
    │
    ▼
Deployment
```

---

# 💻 Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-learn

## Deep Learning

- PyTorch
- Hugging Face Transformers

## NLP

- Transformers
- Tokenizers

## Web Framework

- Streamlit

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

---

# 🌐 Web Application

TruthLens provides a user-friendly interface where users can:

- Paste a news article
- Analyze its credibility
- View prediction confidence
- Receive instant AI-generated results

The application is designed for easy deployment on Streamlit Cloud.

---

# 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

These metrics ensure reliable fake news detection performance.

---

# 📂 Repository Structure

```
├── interface/
│   ├── app.py
│   ├── requirements.txt
│
├── models/
│   ├── trained_model
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│
├── dataset/
│
├── README.md
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/TruthLens.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🌍 Applications

TruthLens can be used in:

- News Verification Platforms
- Journalism
- Fact-Checking Organizations
- Social Media Monitoring
- Government Agencies
- Educational Institutions
- Research Organizations
- Digital Media Platforms

---

# 🔮 Future Enhancements

Future improvements include:

- Integration with live news APIs
- Automatic web article extraction from URLs
- Explainable AI visualizations (LIME/SHAP)
- Source credibility analysis
- Knowledge Graph-based fact verification
- Retrieval-Augmented Generation (RAG) for evidence retrieval
- Real-time misinformation monitoring
- Browser extension for instant news verification
- Mobile application deployment

---

# 📚 Skills Demonstrated

This project showcases expertise in:

- Natural Language Processing (NLP)
- Large Language Models (LLMs)
- Transformer Models
- Hugging Face
- Machine Learning
- Deep Learning
- Streamlit
- Explainable AI
- Data Preprocessing
- Model Deployment
- Python Development

---

# 👥 Contributors

Developed as a collaborative AI project focused on leveraging Natural Language Processing and Machine Learning to detect fake news and improve digital information credibility.

---

# ⭐ Conclusion

TruthLens demonstrates the practical application of transformer-based Natural Language Processing for multilingual fake news detection. By combining AI-powered text classification, language processing, confidence scoring, and an intuitive web interface, the system provides an effective solution for combating misinformation. The project highlights the potential of AI-assisted fact-checking systems in supporting journalists, researchers, and the general public in identifying credible information in today's rapidly evolving digital landscape.
