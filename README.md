Spam Detection using Machine Learning (Naive Bayes)
Project Overview

This project implements a spam detection system using Natural Language Processing and Machine Learning techniques. The goal is to classify SMS messages as either Spam or Ham (Not Spam). The model is trained using the Multinomial Naive Bayes algorithm and provides accurate predictions based on text content.

The project covers the complete machine learning pipeline including data preprocessing, feature extraction, model training, evaluation, and saving trained models for future predictions.

Objectives
To automatically detect spam messages
To apply NLP preprocessing techniques
To build and evaluate a machine learning classification model
To save trained models for reuse
To implement a reusable prediction function

Technologies Used
1. Python
2. Pandas
3. NumPy
4. NLTK
5. Scikit-learn
6. Matplotlib
7. Joblib
8. VS Code
9. Jupyter Notebook

Project Structure
1 spam-detection-ml
1.1 dataset
1.1.1sms.tsv

2 model
2.1 spam_model.pkl
2.2 tfidf_vectorizer.pkl

3 notebook
3.1 spam_detection.ipynb

 4 README.md

Workflow
Load dataset
Data cleaning and preprocessing
Feature extraction using TF-IDF
Model training using Naive Bayes
Model evaluation
Save trained model
Predict spam messages
Model Performance
Accuracy achieved by the trained model is approximately 97.4 percent using the Multinomial Naive Bayes algorithm.

How to Run
Step 1 Install Required Libraries
pip install pandas numpy scikit-learn nltk matplotlib joblib

Step 2 Run the Notebook
Open and execute the notebook file
notebook/spam_detection.ipynb

Sample Prediction
predict_spam("Congratulations! You have won a free prize. Click now!")

Output

spam

Saved Models

spam_model.pkl contains the trained classification model
tfidf_vectorizer.pkl contains the trained TF-IDF vectorizer
These files allow prediction without retraining the model.

Future Scope

Deploy the project as a web application using Flask or Streamlit
Improve accuracy using deep learning models
Add real time SMS classification interface

