# 📰 Fake News Detection using Machine Learning

A Machine Learning and Natural Language Processing (NLP) based web application that classifies news articles as **Real** or **Fake** using **TF-IDF Vectorization** and a **Passive Aggressive Classifier**.

The trained Machine Learning model is integrated with a **Flask web application** to provide real-time predictions through a simple and user-friendly interface.

---

## 📋 Table of Contents

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
- [Installation](#-installation)
- [Usage](#-usage)
- [Visualizations](#-visualizations)
- [Future Improvements](#-future-improvements)
- [Applications](#-applications)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)
- [Support](#-support)
- [License](#-license)

---

## 🚀 Introduction

Fake news and misinformation have become significant challenges in the digital information ecosystem.

This project demonstrates how Machine Learning and Natural Language Processing can be used to analyse textual news content and classify it as **Real** or **Fake**.

The system processes news text, extracts important features using **TF-IDF**, and uses a **Passive Aggressive Classifier** to generate the prediction.

The trained model is integrated with a **Flask backend**, allowing users to submit news content and receive a classification result in real time.

> **Note:** This system provides a machine-learning classification and should not be considered a replacement for independent fact-checking.

---

## 🎯 Problem Statement

The objective of this project is to develop an automated system that can classify news articles based on their textual characteristics.

### Input

The system processes:

- News headline
- Author
- Article body

### Output

The system predicts one of the following classes:

- ✅ **REAL**
- ❌ **FAKE**

---

## 🎯 Objectives

The main objectives of this project are:

- Collect and preprocess a labelled news dataset.
- Perform Natural Language Processing on news text.
- Convert textual data into numerical features using TF-IDF.
- Train a Passive Aggressive Classifier.
- Evaluate the trained Machine Learning model.
- Save the trained model and vectorizer.
- Develop a Flask backend for real-time prediction.
- Create a simple and user-friendly web interface.
- Demonstrate an end-to-end Machine Learning deployment workflow.

---

## ✨ Features

- 📰 Real and fake news classification
- 🤖 Machine Learning based prediction
- 🧠 Natural Language Processing
- 🔤 TF-IDF feature extraction
- ⚡ Passive Aggressive Classifier
- 🌐 Flask web application
- 📊 Model evaluation
- 📈 Confusion matrix
- 💾 Trained model serialization
- 🔄 Real-time prediction
- 🖥️ User-friendly web interface

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Passive Aggressive Classifier
- TF-IDF Vectorizer

### Natural Language Processing

- Text preprocessing
- Text cleaning
- Feature extraction
- TF-IDF

### Backend

- Flask

### Frontend

- HTML
- CSS
- JavaScript

### Data Processing

- Pandas
- NumPy

### Development Tools

- Jupyter Notebook
- Git
- GitHub
- VS Code

---

## 🏗️ Project Architecture

The complete system follows this workflow:

**News Dataset**  
↓  
**Data Preprocessing**  
↓  
**Text Processing**  
↓  
**TF-IDF Vectorization**  
↓  
**Passive Aggressive Classifier**  
↓  
**Model Training**  
↓  
**Saved Model and Vectorizer**  
↓  
**Flask Backend**  
↓  
**Web Interface**  
↓  
**REAL / FAKE Prediction**

---

## 📂 Project Structure

The repository contains the following main components:

- 📁 **Images** — Block diagram, process flow and confusion matrix
- 📁 **dataset** — Training and testing datasets
- 📁 **static** — CSS, JavaScript and UI assets
- 📁 **templates** — HTML templates for the web application
- 📓 **Fake_News_Detector-PA.ipynb** — Jupyter Notebook for preprocessing, analysis and model training
- 🐍 **app.py** — Flask application
- 🤖 **model.pkl** — Trained Passive Aggressive Classifier
- 🔤 **vector.pkl** — Trained TF-IDF Vectorizer
- 📦 **requirements.txt** — Python dependencies
- 📄 **README.md** — Project documentation

---

## 📊 Dataset

The project uses a labelled news dataset containing real and fake news articles.

### Training Dataset

The training dataset is stored in:

`dataset/train.csv`

Important attributes include:

| Column | Description |
|---|---|
| `id` | Unique identifier |
| `title` | News headline |
| `author` | Article author |
| `text` | Main article content |
| `label` | Classification label |

### Label Encoding

| Label | Meaning |
|---|---|
| `0` | Real / Reliable |
| `1` | Fake / Unreliable |

### Testing Dataset

The testing dataset is stored in:

`dataset/test.csv`

It contains news articles used for testing and generating predictions.

---

## 🔄 Machine Learning Workflow

### 1. Data Collection

The labelled news dataset is loaded using Pandas.

### 2. Data Preprocessing

The textual data is cleaned and prepared for Machine Learning.

This includes:

- Handling missing values
- Preparing news text
- Cleaning unnecessary information
- Preparing text for feature extraction

### 3. Feature Extraction

The project uses **TF-IDF Vectorization** to convert textual information into numerical feature vectors.

### 4. Model Training

The transformed text features are provided to the **Passive Aggressive Classifier** for training.

### 5. Model Evaluation

The trained model is evaluated using classification metrics and a confusion matrix.

### 6. Model Serialization

The trained classifier is stored in:

`model.pkl`

The trained TF-IDF vectorizer is stored in:

`vector.pkl`

### 7. Deployment

The saved model and vectorizer are loaded into the Flask application.

### 8. Prediction

The user enters news content through the web application and receives the predicted classification.

---

## 🤖 Model

### Passive Aggressive Classifier

The project uses the **Passive Aggressive Classifier**, an online learning algorithm suitable for text classification.

The classifier updates its parameters when a prediction is incorrect while remaining relatively unchanged when predictions are correct.

### TF-IDF Vectorizer

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It converts text into numerical feature vectors based on the importance of words within the dataset.

### Prediction Pipeline

**News Text → Text Preprocessing → TF-IDF → Passive Aggressive Classifier → Prediction**

---

## 📈 Performance

The Passive Aggressive Classifier achieved approximately:

### **~96% Accuracy**

on the validation data used during project development.

The project also includes a **Confusion Matrix** to visualize classification results.

Actual performance may vary depending on dataset splitting, preprocessing and model configuration.

---

## 🌐 Web Application

The project includes a Flask-based web application for real-time prediction.

### Application Workflow

**User Input**  
↓  
**Flask Backend**  
↓  
**Text Processing**  
↓  
**TF-IDF Vectorization**  
↓  
**Passive Aggressive Classifier**  
↓  
**Prediction**  
↓  
**REAL / FAKE**

The Flask backend is implemented in:

`app.py`

---

## ⚙️ Installation

### 1. Clone the Repository

Use the following command:

`git clone https://github.com/prashant-00-22/Fake-News-Detection-using-Machine-Learning.git`

### 2. Navigate to the Project

`cd Fake-News-Detection-using-Machine-Learning`

### 3. Create a Virtual Environment

Windows:

`python -m venv my_env`

Activate the environment:

`.\my_env\Scripts\Activate.ps1`

### 4. Install Dependencies

`pip install -r requirements.txt`

If `requirements.txt` is not available:

`pip install flask pandas numpy scikit-learn`

---

## ▶️ Usage

Start the Flask application:

`python app.py`

After the application starts, open:

`http://127.0.0.1:5000/`

You can also use:

`http://localhost:5000/`

Enter the news content into the web application and submit it for classification.

The application will return:

**REAL**

or

**FAKE**

---

## 🖼️ Visualizations

The repository contains visual documentation and model evaluation images.

### Block Diagram

`Images/BlockDiagram.jpg`

### Process Flow Diagram

`Images/Processflow.jpg`

### Confusion Matrix

`Images/ConfusionMatrix.jpg`

---

## 📸 Screenshots

Screenshots of the application can be added to this section.

For example:

`Images/LandingPage.png`

`Images/PredictionPage.png`

Make sure the filenames match the actual files present in the repository.

---

## 🔮 Future Improvements

The project can be further enhanced with:

- 🤖 BERT and Transformer-based models
- 🧠 Deep Learning based classification
- 🌍 Multilingual fake news detection
- 🔗 URL-based news extraction
- 📰 Real-time news API integration
- 📊 Prediction confidence score
- 🔍 Explainable AI
- 👤 User authentication
- 🗃️ Prediction history
- 🐳 Docker deployment
- ☁️ Cloud deployment
- 🔐 Secure REST API

---

## 💡 Applications

Potential applications of the system include:

- Content moderation
- News analysis
- Social media monitoring
- Misinformation screening
- Educational NLP projects
- Machine Learning research
- Automated text classification
- Information quality analysis

The system is intended as a **screening and classification tool**, not as a definitive fact-checking system.

---

## 📚 Learning Outcomes

This project provided practical experience with:

- Machine Learning
- Natural Language Processing
- Text Classification
- TF-IDF Feature Extraction
- Passive Aggressive Classification
- Model Evaluation
- Confusion Matrix
- Flask Development
- Python Programming
- Model Serialization
- Git and GitHub
- Machine Learning Deployment

---

## 👨‍💻 Author

### Prashant

**GitHub:**  
https://github.com/prashant-00-22

**Email:**  
prashantsharma0422@gmail.com

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Your support and feedback are appreciated!

---

## 📄 License

This project is developed for **educational, learning, and development purposes**.

---

## 🚀 Project Summary

This project demonstrates how a Machine Learning and NLP-based application can be designed, developed, integrated, tested and deployed as a complete web-based system.
