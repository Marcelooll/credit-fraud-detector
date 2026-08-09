# FraudSentinel
https://credit-fraud-detector-jsyhjdnslzpjchs5rn5wew.streamlit.app/

---

FraudSentinel is a complete end-to-end machine learning project for credit card fraud anomaly detection. It combines data loading, feature engineering, unsupervised model training, and a polished interactive dashboard built with Streamlit.

This project is designed to be both a strong portfolio piece and a practical example of how to build an anomaly detection system that can be explained to recruiters, managers, or technical interviewers.

Version 1.1 includes a multilingual Streamlit interface, live transaction simulation, batch CSV analysis, explanatory diagnostics, and deployment-ready compatibility for Streamlit Cloud.

If you find a bug, have a suggestion, or want a feature added, please open an issue or contact the maintainer.

---

## 1. Streamlit Cloud Deployment

This project is ready to be deployed on Streamlit Cloud with the standard structure:

- root entrypoint: app.py
- dependencies: requirements.txt
- Python runtime: runtime.txt
- Streamlit config: .streamlit/config.toml

Use this deployment URL after publishing the repository:

https://share.streamlit.io/Marcelooll/credit-fraud-detector/main/app.py

## 2. Project Overview

FraudSentinel uses an Isolation Forest model to detect suspicious transactions without requiring labeled fraud examples during training. The system is trained on a real-world credit card fraud dataset and exposes its results through a web app that supports:

- live transaction simulation;
- batch CSV analysis;
- anomaly scoring;
- human-readable explanations;
- interactive visualizations.

The project is a good example of a real-world ML workflow because it includes the full lifecycle:

1. data acquisition;
2. preprocessing and feature engineering;
3. model training;
4. inference;
5. presentation in an easy-to-use interface.

---

## 2. Why this project is interesting

This project is valuable because it demonstrates several core competencies:

- supervised vs. unsupervised learning concepts;
- feature engineering from raw transaction data;
- anomaly detection using Isolation Forest;
- deployment of a data science project as a web application;
- communication of predictions through an intuitive interface.

For a student or junior professional, this is much stronger than simply showing a notebook because it shows that you can build a functional product, not just experiment with code.

---

## 3. Main Features

- Real-time simulation of transaction risk
- Batch inference from uploaded CSV files
- Visual analytics using Plotly
- Rule-based explanation layer for each prediction
- Session history and CSV export
- Modern and responsive Streamlit interface
- Model artifacts persisted locally for fast reuse

---

## 4. Technologies Used

### Core stack

- Python 3.10+
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for Isolation Forest and preprocessing
- Joblib for model serialization
- Streamlit for the interactive web app
- Plotly for charts and dashboards

### Data and ML support

- KaggleHub for dataset download
- StandardScaler for feature normalization
- IsolationForest for anomaly detection

### Optional / experimental dependencies

- SHAP
- imbalanced-learn

These libraries are included to support experimentation and future model analysis, but the core application works with the main stack above.

---

## 5. Architecture

The system follows a simple and modular structure:

```text
Raw transaction data
        │
        ▼
Feature engineering
        │
        ▼
Preprocessing and scaling
        │
        ▼
Isolation Forest training
        │
        ▼
Model artifacts (.pkl files)
        │
        ▼
Streamlit dashboard
        ├── Live prediction
        ├── Batch CSV processing
        └── Visualization and explanation layer
```

### Components

- train.py: prepares the data, engineers features, trains the model, evaluates it, and saves the artifacts.
- app.py: loads the trained model, accepts user input or CSV files, performs inference, and renders the dashboard.
- model/: stores the trained model, scaler, and feature list.
- requirements.txt: contains the Python dependencies needed to run the project.

---

## 6. Project Structure

```text
credit-fraud-detector/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── README.pt-BR.md
└── model/
    ├── isolation_forest.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

---

## 7. Prerequisites

Before running the project locally, make sure you have:

- Python 3.10 or higher
- pip installed
- a Kaggle account
- Kaggle API credentials configured

### Configure Kaggle credentials

1. Go to https://www.kaggle.com/settings
2. Open the API section
3. Create a new API token
4. Save the downloaded kaggle.json file in:
   - Windows: C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
   - macOS/Linux: ~/.kaggle/kaggle.json

If the credentials are not configured correctly, the training script will not be able to download the dataset.

---

## 8. Installation and Setup

### Windows

```bash
cd credit-fraud-detector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd credit-fraud-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 9. How to Run the Project

### Step 1: Train the model

```bash
python train.py
```

This script will:

- download the dataset from Kaggle;
- engineer new features such as hour, day of week, age, and log-transformed amount;
- scale the input data;
- train the Isolation Forest model;
- save the artifacts in the model/ directory.

### Step 2: Start the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 10. How to Use the Application

### Live simulation tab

Use this mode to test one transaction at a time.

You can adjust values such as:

- transaction amount;
- hour of transaction;
- day of week;
- cardholder age;
- city population;
- merchant coordinates;
- cardholder coordinates.

The app will return:

- a fraud-risk score;
- a verdict such as safe or suspicious;
- a human-readable explanation;
- visual diagnostic charts.

### Batch processing tab

Use this tab to upload a CSV file and run inference over many rows at once.

You can:

- upload a custom dataset;
- test with a synthetic sample;
- inspect the results in a table;
- download predictions as CSV.

### Insights tab

This section provides deeper understanding of the model and its behavior, including:

- model parameters;
- anomaly detection theory;
- visualizations of the result distribution.

---

## 11. Model Logic

The project uses an unsupervised anomaly detection approach.

Why Isolation Forest?

- it does not require labeled fraud examples to detect anomalies;
- it is well suited for rare events such as fraud;
- it produces anomaly scores that can be interpreted as a risk signal.

The model outputs a score that indicates how unusual a transaction is compared with the learned baseline. In the app, this is translated into an interpretable risk view for end users.

---

## 12. Deployment on Streamlit Community Cloud

Yes — this project is very suitable for deployment on Streamlit Community Cloud.

### Recommended deployment flow

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the repository and branch.
5. Set the main file to app.py.
6. Deploy.

### Important notes

- The app uses the trained model files located in the model/ folder, so they should be present in the repository before deployment.
- Keep secrets such as Kaggle credentials out of the repository.
- For a production-grade deployment, you may later add a more robust environment management flow.

This makes the project especially attractive for a portfolio because it demonstrates not only model development but also deployment readiness.

---

## 13. Suggested Interview Talking Points

If you want to use this project in interviews or on your resume, these are strong points to highlight:

- built a complete machine learning pipeline from data to UI;
- implemented an unsupervised anomaly detection solution;
- worked with real-world transaction data and feature engineering;
- deployed a data science application with Streamlit;
- created a user-facing experience with explainability and visualization.

---

## 14. Future Improvements

Possible next steps for the project include:

- adding explainability with SHAP;
- improving model calibration;
- adding support for more advanced fraud heuristics;
- integrating a database for historical transactions;
- creating a REST API for model inference;
- adding automated tests.

---

## 15. License

This project is available for educational and personal use.

If you want, you can also adapt it for a more formal commercial or academic use case.
