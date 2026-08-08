"""
train.py - Credit Card Fraud Detection Training Pipeline
=========================================================
Loads the Kaggle dataset, engineers features, trains an Isolation Forest
model, evaluates it and saves both the model and scaler to ./model/.

Usage:
    python train.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

NUMERICAL_FEATURES = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "hour",
    "day_of_week",
    "age",
    "amt_log",
]

CONTAMINATION = 0.01  # ~1% expected fraud rate (Isolation Forest hyperparameter)
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the fraud detection dataset from Kaggle via kagglehub."""
    print("Loading dataset from Kaggle (kartik2112/fraud-detection)...")
    try:
        import kagglehub

        # Download dataset files to local cache and get the folder path
        dataset_path = kagglehub.dataset_download("kartik2112/fraud-detection")

        # Look for the training CSV (fraudTrain.csv preferred, else first CSV found)
        train_csv = os.path.join(dataset_path, "fraudTrain.csv")
        if not os.path.exists(train_csv):
            csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {dataset_path}")
            train_csv = csv_files[0]

        print(f"Reading: {train_csv}")
        df = pd.read_csv(train_csv)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load the Kaggle dataset. "
            f"Make sure you have configured your Kaggle API credentials "
            f"(~/.kaggle/kaggle.json).\nOriginal error: {exc}"
        )

    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and create meaningful derived features."""
    print("\nEngineering features...")

    df = df.copy()

    # Parse transaction datetime
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], errors="coerce"
    )

    # Extract time-based features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=Mon, 6=Sun

    # Customer age (year difference between DOB and transaction year)
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = (
        df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    ).fillna(df["trans_date_trans_time"].dt.year - 1980)

    # Log-transform transaction amount to reduce skew
    df["amt_log"] = np.log1p(df["amt"])

    print(f"    Features ready: {NUMERICAL_FEATURES}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Scale numerical features. Returns X_scaled (all rows) and the scaler."""
    print("\nScaling features...")

    X = df[NUMERICAL_FEATURES].fillna(0).copy()
    scaler = StandardScaler()

    # Fit scaler ONLY on legitimate transactions (unsupervised best practice)
    legitimate_mask = df["is_fraud"] == 0
    scaler.fit(X[legitimate_mask])

    X_scaled = scaler.transform(X)
    print(f"    Scaler fitted on {legitimate_mask.sum():,} legitimate transactions.")
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_scaled: np.ndarray) -> IsolationForest:
    """Train Isolation Forest on legitimate-only data (unsupervised approach)."""
    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    print("Training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model: IsolationForest, X_scaled: np.ndarray, y_true: np.ndarray):
    """
    Evaluate the model.
    Isolation Forest returns +1 (normal) or -1 (anomaly).
    We map -1 → fraud=1 and +1 → fraud=0.
    """
    print("\nEvaluating model...")

    raw_preds = model.predict(X_scaled)          # +1 or -1
    y_pred = np.where(raw_preds == -1, 1, 0)     # 1=fraud, 0=legit

    # Anomaly scores (more negative = more anomalous)
    scores = model.decision_function(X_scaled)
    # Invert so that higher score = more fraudulent (useful for ROC-AUC)
    scores_inverted = -scores

    try:
        roc = roc_auc_score(y_true, scores_inverted)
    except Exception:
        roc = float("nan")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix ---")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n    ROC-AUC Score: {roc:.4f}")
    print("-" * 65)
    return y_pred, scores_inverted


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model: IsolationForest, scaler: StandardScaler):
    """Persist model and scaler to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(NUMERICAL_FEATURES, FEATURE_NAMES_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Credit Card Fraud Detection — Training Pipeline")
    print("=" * 65)

    df = load_data()
    df = engineer_features(df)
    X_scaled, scaler = preprocess(df)
    model = train_model(X_scaled)
    evaluate(model, X_scaled, df["is_fraud"].values)
    save_artifacts(model, scaler)

    print("\nPipeline finished successfully!")
    print("You can now launch the dashboard with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
