📧 Spam Detection using Machine Learning (Naive Bayes)
📌 Project Overview

This project implements a Spam Detection System using Natural Language Processing (NLP) and Machine Learning techniques.
It classifies SMS messages as Spam or Ham (Not Spam) using the Multinomial Naive Bayes algorithm.

The model is trained on a real-world SMS dataset and achieves high accuracy in spam classification.

🎯 Objectives

Detect spam SMS messages automatically

Apply NLP preprocessing techniques

Train and evaluate a machine learning classifier

Save trained models for reuse

Build a reusable prediction function

🧠 Technologies Used

Python

Pandas

NumPy

NLTK

Scikit-learn

Matplotlib

Joblib

VS Code

Jupyter Notebook

📂 Project Structure
spam-detection-ml
│
├── dataset
│   └── sms.tsv
│
├── model
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook
│   └── spam_detection.ipynb
│
└── README.md

⚙️ Workflow

Load Dataset

Data Cleaning & Preprocessing

Feature Extraction using TF-IDF

Model Training using Naive Bayes

Model Evaluation

Save Trained Model

Predict Spam Messages

📊 Model Performance
Metric	Value
Accuracy	97.4%
Algorithm	Multinomial Naive Bayes
🚀 How to Run
Step 1: Install Dependencies
pip install pandas numpy scikit-learn nltk matplotlib joblib

Step 2: Run Notebook

Open and execute:

notebook/spam_detection.ipynb

🔮 Spam Prediction Example
predict_spam("Congratulations! You have won a free prize. Click now!")


Output:

spam

💾 Saved Models

spam_model.pkl → Trained classification model

tfidf_vectorizer.pkl → Text vectorization model

These allow direct prediction without retraining.

📈 Future Improvements

Deploy as a web application using Flask / Streamlit

Use Deep Learning (LSTM / BERT) for improved accuracy

Add real-time SMS classification interface

👩‍💻 Author

Priyanka Chitranshi
B.Tech CSE Student
GitHub: Priyanka212-alt

⭐ If you like this project, give it a star!

If you want, I can also help you:

✅ Make project report PDF
✅ Create resume-ready description
✅ Build web app version
