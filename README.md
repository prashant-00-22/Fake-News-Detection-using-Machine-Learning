# Fake News Detection using Machine Learning

A Machine Learning and Natural Language Processing (NLP) based web application that classifies news articles as **Real** or **Fake** using **TF-IDF feature extraction** and a **Passive Aggressive Classifier**.

The trained machine learning model is integrated with a **Flask web application** to provide real-time predictions through a simple web interface.

---

## 📌 Table of Contents

- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model](#-model)
- [Performance](#-performance)
- [Web Application](#-web-application)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Example](#-example)
- [Visualizations](#-visualizations)
- [Future Improvements](#-future-improvements)
- [Applications](#-applications)
- [Author](#-author)
- [License](#-license)

---

## 🚀 Introduction

Fake news and misinformation have become major challenges in the digital information ecosystem. News content can spread rapidly through websites and social media platforms, making automated detection systems increasingly useful.

This project implements a **Fake News Detection System** using Machine Learning and Natural Language Processing.

The system processes the textual content of a news article, converts the text into numerical features using **TF-IDF Vectorization**, and then uses a **Passive Aggressive Classifier** to predict whether the news is likely to be:

- ✅ **Real**
- ❌ **Fake**

The trained model is deployed through a **Flask web application**, allowing users to enter news content and receive a prediction in real time.

> **Note:** The prediction is a machine-learning classification result and should not be treated as definitive verification of the factual accuracy of a news article.

---

## 🎯 Problem Statement

The objective of this project is to develop a machine learning system capable of classifying news articles based on their textual characteristics.

The system learns patterns from previously labelled news data and uses these patterns to classify unseen news content.

### Input

News article text containing information such as:

- Headline
- Author
- Article body

### Output

The system predicts one of the following classes:

```text
REAL
FAKE
🎯 Objectives

The main objectives of this project are:

Collect and preprocess a labelled news dataset.
Perform Natural Language Processing on news text.
Convert textual data into numerical features using TF-IDF.
Train a Passive Aggressive Classifier.
Evaluate the trained model.
Save the trained model and vectorizer.
Develop a Flask backend for prediction.
Create a web interface for real-time classification.
Provide a simple and user-friendly prediction workflow.
✨ Features
📰 Fake and real news classification
🤖 Machine Learning based prediction
🧠 Natural Language Processing
🔤 TF-IDF text feature extraction
⚡ Passive Aggressive Classifier
🌐 Flask web application
📊 Model evaluation
📈 Confusion matrix visualization
💾 Serialized ML model using pickle
🔄 Real-time prediction
🖥️ Simple web interface
🛠️ Technology Stack
Programming Language
Python
Machine Learning
Scikit-learn
Passive Aggressive Classifier
TF-IDF Vectorizer
Natural Language Processing
Text preprocessing
Tokenization
Stop-word handling
TF-IDF feature extraction
Backend
Flask
Frontend
HTML
CSS
JavaScript
Development Tools
Jupyter Notebook
Git
GitHub
VS Code
Data Processing
Pandas
NumPy
🏗️ Project Architecture

The overall workflow of the project is:

                    ┌─────────────────────┐
                    │     News Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Preprocessing│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Processing   │
                    │        +            │
                    │  TF-IDF Vectorizer  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Passive Aggressive  │
                    │     Classifier      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Model Training   │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
              ┌─────────────┐     ┌─────────────┐
              │  model.pkl  │     │ vector.pkl  │
              └──────┬──────┘     └──────┬──────┘
                     │                   │
                     └─────────┬─────────┘
                               ▼
                    ┌─────────────────────┐
                    │    Flask Backend    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Web Interface    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  REAL / FAKE Result │
                    └─────────────────────┘
📂 Project Structure
Fake-News-Detection-using-Machine-Learning/
│
├── Images/
│   ├── BlockDiagram.jpg
│   ├── Processflow.jpg
│   └── ConfusionMatrix.jpg
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── Landingpage.html
│   └── prediction page.html
│
├── Fake_News_Detector-PA.ipynb
│
├── app.py
│
├── model.pkl
│
├── vector.pkl
│
├── requirements.txt
│
└── README.md
📊 Dataset

The project uses a labelled news dataset containing real and fake news articles.

Training Dataset

File:

dataset/train.csv

The training dataset contains fields such as:

Column	Description
id	Unique identifier
title	News headline
author	Article author
text	Main article content
label	Target classification label
Label Encoding
1 → Fake / Unreliable
0 → Real / Reliable
Testing Dataset

File:

dataset/test.csv

The testing dataset contains news articles used for generating predictions.

🔄 Machine Learning Workflow

The project follows the following Machine Learning pipeline:

1. Data Collection

The labelled news dataset is loaded using Pandas.

2. Data Exploration

The dataset is analysed to understand:

Number of records
Missing values
Label distribution
Text characteristics
Dataset structure
3. Data Preprocessing

The textual data is prepared for machine learning.

Typical preprocessing includes:

Handling missing values
Combining relevant text fields
Cleaning textual data
Removing unnecessary information
Preparing text for vectorization
4. Feature Extraction

The cleaned text is converted into numerical features using:

TF-IDF Vectorizer

TF-IDF assigns numerical importance to words based on their occurrence within the dataset.

5. Model Training

The project uses:

PassiveAggressiveClassifier

The classifier is trained using the TF-IDF transformed text data.

6. Model Evaluation

The trained model is evaluated using classification performance metrics and a confusion matrix.

7. Model Serialization

The trained classifier is saved as:

model.pkl

The TF-IDF vectorizer is saved as:

vector.pkl
8. Deployment

The saved model and vectorizer are loaded into the Flask application.

9. Prediction

Users submit news content through the web interface, and the Flask application returns the predicted class.

🤖 Model
Passive Aggressive Classifier

The project uses a Passive Aggressive Classifier, an online learning algorithm that is well suited to large-scale text classification tasks.

The classifier updates its model when a prediction is incorrect while remaining relatively unchanged when predictions are correct.

TF-IDF Vectorizer

The Term Frequency-Inverse Document Frequency (TF-IDF) technique converts textual data into numerical feature vectors.

The general idea is:

Raw News Text
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Numerical Feature Vector
      ↓
Passive Aggressive Classifier
      ↓
Prediction
📈 Performance

The Passive Aggressive Classifier achieved approximately:

~96% Accuracy

on the validation data used during project development.

Evaluation

The project includes a confusion matrix to analyse:

True Positives
True Negatives
False Positives
False Negatives

The evaluation results depend on the dataset split, preprocessing, and model configuration used during training.

🌐 Web Application

The Flask application provides a simple interface for submitting news content.

Application Workflow
User
 │
 ▼
Enter News Content
 │
 ▼
Flask Application
 │
 ▼
TF-IDF Vectorizer
 │
 ▼
Passive Aggressive Classifier
 │
 ▼
Prediction
 │
 ├── REAL
 │
 └── FAKE

The Flask application is implemented in:

app.py
📋 Prerequisites

Before running the project, make sure the following are installed:

Python 3.8 or higher
Git
pip
Virtual environment support

You can verify Python installation using:

python --version

Verify pip:

pip --version
⚙️ Installation
1. Clone the Repository
git clone https://github.com/prashant-00-22/Fake-News-Detection-using-Machine-Learning.git

Move into the project directory:

cd Fake-News-Detection-using-Machine-Learning
2. Create a Virtual Environment
Windows PowerShell
python -m venv my_env

Activate the environment:

.\my_env\Scripts\Activate.ps1

If PowerShell blocks script execution, you can use Command Prompt:

my_env\Scripts\activate
3. Install Dependencies

Install all required Python packages:

pip install -r requirements.txt

If requirements.txt is not available, install the main dependencies:

pip install flask pandas numpy scikit-learn
▶️ Running the Application

After installing the dependencies, run:

python app.py

If the application starts successfully, Flask will display a local server address similar to:

http://127.0.0.1:5000/

Open the address in your browser.

You can also use:

http://localhost:5000/
🧪 Example
Input
Scientists have announced a new research finding...

The entered text is processed by the application.

Processing
Input News
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Passive Aggressive Classifier
    ↓
Prediction
Output
Prediction: REAL

or

Prediction: FAKE

The result represents the classification produced by the trained model.

🖼️ Visualizations

The repository contains several project diagrams and evaluation visualizations.

Block Diagram
Images/BlockDiagram.jpg

The block diagram represents the overall system architecture.

Process Flow
Images/Processflow.jpg

The process flow illustrates the steps involved in processing and classifying news.

Confusion Matrix
Images/ConfusionMatrix.jpg

The confusion matrix provides a visual representation of model classification results.

📸 Project Screenshots

You can add screenshots of the web application here.

Example:

![Landing Page](Images/LandingPage.png)

![Prediction Page](Images/PredictionPage.png)

![Confusion Matrix](Images/ConfusionMatrix.jpg)

Replace the filenames with the actual screenshot filenames available in your repository.

🔮 Future Improvements

The project can be further enhanced with:

Deep Learning based text classification
Transformer models such as BERT
Advanced NLP preprocessing
Multilingual fake news detection
News source verification
URL-based article extraction
Real-time news API integration
Explainable AI for prediction reasoning
Confidence score for predictions
REST API deployment
Docker containerization
Cloud deployment
Database integration
User authentication
Prediction history
💡 Applications

Potential applications of the system include:

Content moderation systems
Social media monitoring
News verification platforms
Educational NLP projects
Research in misinformation detection
Automated content analysis
Information quality monitoring

The model should be considered a screening/classification tool rather than a replacement for independent fact-checking.

📚 Learning Outcomes

Through this project, the following concepts were explored:

Machine Learning
Natural Language Processing
Text Classification
TF-IDF Feature Extraction
Passive Aggressive Classification
Model Evaluation
Confusion Matrix
Flask Application Development
REST-style backend integration
Python development
Git and GitHub
Machine Learning model serialization
👨‍💻 Author
Prashant

GitHub:
https://github.com/prashant-00-22

Email:
prashantsharma0422@gmail.com

📄 License

This project is intended for educational, learning, and demonstration purposes.

You may modify and extend the project for academic and personal development.
