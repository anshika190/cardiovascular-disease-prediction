# 🫀 Cardiovascular Disease Prediction & Health Analytics System (CardioSense)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask%20%7C%20Streamlit-orange.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20(MLP%20Neural%20Network)-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)]()

A comprehensive, end-to-end Machine Learning healthcare platform for assessing cardiovascular disease (CVD) risk and performing interactive population health analytics.

The project incorporates two integrated components:
1. **CardioSense Clinical Predictor (Flask Web Application)**: An interactive clinical risk prediction tool powered by a deep Multi-Layer Perceptron (MLP) Neural Network that estimates real-time CVD probabilities, visualizes risk factors, and provides actionable lifestyle recommendations.
2. **Health Analytics & EDA Dashboard (Streamlit Application)**: An exploratory data analysis suite enabling health professionals to filter cohorts, evaluate BMI and chronic disease distributions, inspect correlation heatmaps, and export customized health datasets.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technologies & Libraries Used](#-technologies--libraries-used)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Machine Learning Model Details](#-machine-learning-model-details)
- [Installation & Setup Guide](#-installation--setup-guide)
- [How to Run Locally](#-how-to-run-locally)
- [API Documentation](#-api-documentation)
- [Deployment Guide](#-deployment-guide)
- [College Deliverables & Submission Info](#-college-deliverables--submission-info)

---

## 🩺 Project Overview
Cardiovascular disease remains the leading cause of mortality globally. Early detection and proactive lifestyle intervention can drastically reduce fatal cardiac events. 

This project analyzes patient lifestyle factors (dietary habits, exercise, smoking history, alcohol consumption), medical history (diabetes, depression, arthritis, cancer), and clinical metrics (BMI, age category, checkup regularity) to:
- Predict individualized probability of cardiovascular disease using artificial neural networks.
- Highlight specific contributing risk factors (e.g., obesity, high fried food intake, lack of physical activity).
- Deliver personalized clinical and lifestyle recommendations.
- Enable macro-level exploratory epidemiological analysis across large health cohorts.

---

## ✨ Key Features

### 1. CardioSense Clinical Risk Predictor (`app.py` - Flask)
- **Interactive Medical Assessment Form**: Structured input fields for demographic, clinical, dietary, and behavioral variables.
- **Dynamic Animated Gauge**: Real-time radial probability meter displaying 0–100% CVD risk score.
- **Risk Categorization**: Clear 4-tier risk classification:
  - 🟢 **Low Risk** (<25%)
  - 🟡 **Moderate Risk** (25%–49%)
  - 🟠 **High Risk** (50%–74%)
  - 🔴 **Very High Risk** (≥75%)
- **Feature Contribution Breakdown**: Visual indicator chips detailing which attributes contributed to increased risk or served as protective factors.
- **Tailored Medical & Lifestyle Advice**: Clinical guidance tailored specifically to the user's predicted risk tier.

### 2. Population Health Analytics Dashboard (`dashboard.py` - Streamlit)
- **Multi-Factor Cohort Slicing**: Dynamic sidebar filters for Age Category, Sex, General Health, and 6 chronic co-morbidities.
- **Cohort Summary & Descriptive Statistics**: Automated KPI summaries and alert banners for high-risk individuals.
- **Distribution Visualizations**: Interactive histograms, KDE curves, and categorical breakdowns.
- **Correlation Heatmap**: Correlation matrix analyzing interactions between height, weight, BMI, and dietary consumption patterns.
- **Dataset Export**: One-click download of dynamically filtered cohort datasets in CSV format.

---

## 🛠️ Technologies & Libraries Used

| Layer | Technologies |
|---|---|
| **Programming Language** | Python 3.10+ |
| **Web Application Backend** | Flask 3.x, Jinja2, Werkzeug, Gunicorn |
| **Data Analytics Dashboard** | Streamlit |
| **Machine Learning & Modeling** | Scikit-Learn (`MLPClassifier`, `StandardScaler`, `LabelEncoder`) |
| **Data Processing & Math** | NumPy, Pandas |
| **Data Visualization** | Matplotlib, Seaborn |
| **Frontend UI / UX** | Semantic HTML5, Modern CSS3 (Glassmorphism / Dark Mode theme), Vanilla JS (Fetch API, SVG Gauge) |
| **Dataset Source** | CDC Behavioral Risk Factor Surveillance System (BRFSS) |

---

## 📂 Project Architecture & Directory Structure

```text
cardiovascular-disease-prediction/
├── app.py                     # Main Flask web application (CardioSense predictor)
├── dashboard.py               # Streamlit interactive exploratory dashboard
├── train_model.py             # Machine learning data generation & MLP training script
├── requirements.txt           # Unified project Python dependencies
├── run_app.bat                # Windows one-click launcher for Flask web app
├── run_dashboard.bat          # Windows one-click launcher for Streamlit dashboard
├── .gitignore                 # Standard Git exclusion file
├── README.md                  # Complete project documentation
│
├── model/                     # Trained machine learning artifacts
│   ├── mlp_model.pkl          # Trained Multi-Layer Perceptron neural network
│   ├── scaler.pkl             # Fitted StandardScaler for numerical feature normalization
│   ├── encoders.pkl           # Categorical label encoders & option mappings
│   └── data.csv               # Model training dataset
│
├── templates/                 # Frontend templates
│   └── index.html             # CardioSense dark-themed responsive user interface
│
├── data/                      # Health data store
│   └── health_dataset.csv     # BRFSS epidemiological dataset for Streamlit analytics
│
├── notebooks/                 # Data science research
│   └── Cardiovascular_Disease_Prediction.ipynb  # Exploratory Data Analysis & experiments
│
└── docs/                      # College project submission documents
    ├── CardiovascularDiseasePrediction_Presentation.pptx # Project presentation slides
    ├── Project_Synopsis.docx   # Formal project synopsis document
    ├── Final_Project_Report.docx# Comprehensive academic project report
    └── Project_Report.pdf      # PDF version of project report
```

---

## 🧠 Machine Learning Model Details

- **Algorithm**: Multi-Layer Perceptron (MLP) Artificial Neural Network
- **Architecture**: `Input (16) → Hidden Layer 1 (128 neurons, ReLU) → Hidden Layer 2 (64 neurons, ReLU) → Hidden Layer 3 (32 neurons, ReLU) → Output Layer (Sigmoid)`
- **Optimization**: Adam Optimizer (`learning_rate_init=0.001`, `max_iter=500`)
- **Class Balancing**: Resampling / minority oversampling to handle real-world clinical class imbalances
- **Preprocessing**: Robust Label Encoding for discrete nominal fields + standard z-score normalization (`StandardScaler`) for continuous numerical predictors.

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your system.
- Git (optional, for cloning).

### 2. Install Required Dependencies
Open your terminal or PowerShell in the project root directory and run:

```bash
pip install -r requirements.txt
```

---

## 💻 How to Run Locally

### Option A: Using Windows One-Click Launchers
- **Launch CardioSense Web App**: Double-click [`run_app.bat`](run_app.bat)
- **Launch Analytics Dashboard**: Double-click [`run_dashboard.bat`](run_dashboard.bat)

---

### Option B: Using Terminal Commands

#### 1. Run the CardioSense Predictor (Flask Web App)
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

#### 2. Run the Health Analytics Dashboard (Streamlit)
```bash
python -m streamlit run dashboard.py
```
Your browser will open automatically at:
```
http://localhost:8501
```

#### 3. (Optional) Retrain the Machine Learning Model
If you wish to re-train the neural network from scratch:
```bash
python train_model.py
```
This will regenerate `model/mlp_model.pkl`, `model/scaler.pkl`, and `model/encoders.pkl` and print the training evaluation metrics.

---

## 🔌 API Documentation

### `POST /predict`
Processes an individual's health factors and returns probability, risk level, contributing factors, and clinical recommendations.

#### Request Body (`application/json`):
```json
{
  "general_health": "Good",
  "checkup": "Within the past year",
  "exercise": "1",
  "sex": "1",
  "age_category": "65-69",
  "bmi": "28.4",
  "smoking": "1",
  "alcohol": "2",
  "fruit": "2",
  "green_veg": "3",
  "fried_potato": "1",
  "diabetes": "0",
  "depression": "0",
  "arthritis": "0",
  "skin_cancer": "0",
  "other_cancer": "0"
}
```

#### Response (`application/json`):
```json
{
  "success": true,
  "probability": 34.2,
  "prediction": 0,
  "risk_level": "Moderate",
  "risk_color": "#f39c12",
  "advice": [
    "Consult your doctor for a cardiovascular assessment.",
    "Aim for at least 150 min of moderate exercise per week.",
    "Reduce fried and processed food intake.",
    "Monitor your blood pressure and cholesterol levels."
  ],
  "contributions": [
    {"label": "Age (55+)", "impact": "high", "direction": "risk"},
    {"label": "BMI 28.4 (Overweight)", "impact": "medium", "direction": "risk"},
    {"label": "Smoking History", "impact": "high", "direction": "risk"},
    {"label": "Regular Exercise", "impact": "medium", "direction": "protective"}
  ]
}
```

---

## 🌐 Deployment Guide

### Deploying CardioSense to Render.com (Recommended Free Hosting)
1. Push this project repository to your GitHub account.
2. Go to [Render.com](https://render.com/) and click **New + > Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**. Once built, Render will provide a public live URL (e.g., `https://cardiosense-cvd.onrender.com`).

### Deploying the Analytics Dashboard to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository, set the main file path to `dashboard.py`, and click **Deploy**.

---

## 🎓 College Deliverables & Submission Info

- **Project Title**: Cardiovascular Disease Risk Prediction and Health Analytics System
- **Domain**: Machine Learning, Healthcare Informatics, Data Analytics & Full-Stack Web Development
- **Student Documentation**:
  - Project Presentation: [`docs/CardiovascularDiseasePrediction_Presentation.pptx`](docs/CardiovascularDiseasePrediction_Presentation.pptx)
  - Project Synopsis: [`docs/Project_Synopsis.docx`](docs/Project_Synopsis.docx)
  - Final Comprehensive Report: [`docs/Final_Project_Report.docx`](docs/Final_Project_Report.docx) & [`docs/Project_Report.pdf`](docs/Project_Report.pdf)
  - Research Notebook: [`notebooks/Cardiovascular_Disease_Prediction.ipynb`](notebooks/Cardiovascular_Disease_Prediction.ipynb)
- **Live Demo Instructions**: Provide teacher with the running localhost URLs or the deployed cloud link along with the presentation slides.
