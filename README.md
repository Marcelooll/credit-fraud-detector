# FraudSentinel — Enterprise Anomaly Detection Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?style=for-the-badge&logo=plotly)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**FraudSentinel** is an enterprise-grade, unsupervised machine learning anomaly detection engine engineered for real-time and high-throughput batch financial transaction auditing. Built upon an optimized **Isolation Forest** architecture trained on over **1.29 million real-world transactions**, FraudSentinel isolates emerging fraud patterns and zero-day financial anomalies without relying on scarce, delayed, or heavily imbalanced ground-truth labels.

---

## Architecture & Data Flow

```text
                                  ┌────────────────────────────────┐
                                  │   Raw Transaction Telemetry    │
                                  │   (Streaming or Batch CSV)     │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │  Vectorized Feature Pipeline   │
                                  │  - Geodesic Coordinate Delta   │
                                  │  - Non-Linear Amount Log (1+x) │
                                  │  - Circadian Temporal Encoding │
                                  │  - Demographic Age Calculation │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │    StandardScaler Transform    │
                                  │   (Fitted on Legit Baseline)   │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │  Isolation Forest (200 iTrees) │
                                  │   Path Length Computation      │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │  Decision Calibration Engine   │
                                  │  - Continuous Risk Score (%)   │
                                  │  - Binary Anomaly Verdict      │
                                  │  - Multidimensional Heuristics │
                                  └───────────────┬────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌────────────────────────────────┐                ┌────────────────────────────────┐
        │   Real-Time Diagnostic UI      │                │   Massive Batch Processor      │
        │   - Risk Gauge & Radar Profile │                │   - High-Volume Analytics      │
        │   - Explainable Diagnostics    │                │   - Statistical Profiling      │
        │   - Persistent Audit Storage   │                │   - Full Export (.CSV)         │
        └────────────────────────────────┘                └────────────────────────────────┘
```

---

## Core Technical Highlights

### 1. Unsupervised Anomaly Isolation
Traditional supervised classifiers suffer from severe label imbalance (often $< 0.5\%$ fraud rate), concept drift, and chargeback reporting latency (frequently 30–90 days). FraudSentinel overcomes these constraints by modeling the topological density of legitimate transactions:
- **Recursive Partitioning**: Subsamples of transaction spaces are partitioned using random binary hyperplanes. Anomalies, possessing distinct behavioral attributes, require significantly shorter tree path lengths $h(x)$ to be isolated.
- **Normalized Anomaly Score**:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
  where $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree (BST).
- **Contamination Calibration**: Tuned at $0.01$ ($1\%$) to maintain an optimal balance between precision and false discovery rate.

### 2. Domain-Specific Feature Engineering
- **Geodesic Delta Calculation**: Measures geographic divergence between the cardholder's residential coordinates $(\text{lat}, \text{long})$ and the transaction terminal coordinates $(\text{merch\_lat}, \text{merch\_long})$.
- **Monetary Log Transformation**: Applies $\log(1 + \text{amt})$ to stabilize high variance and heavy skewness inherent to monetary transaction distributions.
- **Circadian & Day-of-Week Cycle**: Encodes time-of-day ($0\text{--}23\text{h}$) and weekly purchase rhythms to identify nocturnal and atypical spending bursts.
- **Demographic Stratification**: Calculates dynamic cardholder age relative to transaction timestamps and accounts for municipal population density.

---

## Performance & Evaluation Benchmark

Trained and evaluated on the **Kaggle Credit Card Fraud Detection Corpus** ($1,296,675$ records):

| Metric | Result | Context |
| :--- | :--- | :--- |
| **ROC-AUC Score** | **0.90+** | Unsupervised decision boundary ranking |
| **Inference Latency** | **$< 0.8\text{ ms}$ / transaction** | Ultra-low latency single-item evaluation |
| **Batch Throughput** | **$> 12,000\text{ tx/s}$** | Vectorized array inference with NumPy & Scikit-Learn |
| **Training Corpus Scale** | **1,296,675 records** | Scaler fitted strictly on verified legitimate baseline |

---

## Hands-On Testing Playbook (How to Test Each Feature)

To validate the model's inference performance and diagnostic explanations in the web UI:

### Test Case 1: Nominal / Legitimate Transaction (Safe Baseline)
* **Objective**: Confirm that routine daily spending is correctly evaluated as nominal safe behavior.
* **Input Parameters**:
  * **Amount**: `$45.50`
  * **Hour**: `14` (2:00 PM) | **Day of Week**: `Tuesday`
  * **Cardholder Coordinates**: `Lat: 40.7128`, `Long: -74.0060` (New York, NY)
  * **Age**: `35` | **City Population**: `500,000`
  * **Merchant Coordinates**: `Lat: 40.7306`, `Long: -73.9352` (same metropolitan area)
* **Expected Outcome**:
  * **Verdict**: `TRANSACTION VERIFIED // NOMINAL BEHAVIOR (SAFE)`
  * **Risk Score**: `< 25%` (Green Zone)
  * **Raw Decision Score**: `+0.15` to `+0.22`

### Test Case 2: High-Risk Critical Anomaly (Fraud Simulation)
* **Objective**: Force the isolation trees to isolate the vector rapidly under multiple anomalous dimensions.
* **Input Parameters**:
  * **Amount**: `$9,850.00` (severe monetary spike)
  * **Hour**: `03` (critical 3:00 AM nocturnal window) | **Day of Week**: `Sunday`
  * **Cardholder Coordinates**: `Lat: 25.7617`, `Long: -80.1918` (Miami, FL)
  * **Age**: `78` | **City Population**: `120` (isolated rural district)
  * **Merchant Coordinates**: `Lat: 64.2008`, `Long: -149.4937` (Fairbanks, AK — 6,000+ km geodesic leap)
* **Expected Outcome**:
  * **Verdict**: `ANOMALY DETECTADA // ELEVATED RISK (FRAUD)`
  * **Risk Score**: `> 85%` (Red Zone)
  * **Heuristic Diagnostics**: Triggers alerts for monetary spike, nocturnal window, geodesic delta, and low municipal density.

### Test Case 3: High-Throughput Batch Processing
* **Objective**: Evaluate vectorized array inference over bulk transaction streams.
* **Execution Steps**:
  1. Navigate to the `◈ [02] BATCH PROCESSOR` tab.
  2. Click `Test Synthetic Sample (30 records)` or upload a custom CSV dataset.
  3. Inspect real-time KPI metrics, anomaly pie distribution, temporal matrices, and amount vs. risk scatter charts.
  4. Click `⤓ DOWNLOAD COMPLETE CSV PREDICTIONS` to retrieve scored records.

### Test Case 4: Stochastic Monte Carlo Simulation
* **Objective**: Stress-test decision boundary calibration against 50 randomized transaction vectors.
* **Execution Steps**:
  1. Open `◈ [03] INSIGHTS & PARAMETERS`.
  2. Click `Run Monte Carlo Stochastic Test (50 samples)`.
  3. Review the dynamically generated scatter plot illustrating boundary separation between *SAFE* and *FRAUD*.

### Test Case 5: Persistent Audit Telemetry & Timezone Integrity
* **Objective**: Verify that simulated records are stored with standard timezone formatting and persist across page refreshes.
* **Execution Steps**:
  1. Execute a diagnostic in `◈ [01] LIVE SIMULATION`.
  2. Confirm the top row of `◈ Session Audit Log` displays standard formatted timestamps (e.g. `18/08/2026 18:30:00 (UTC-03:00)`).
  3. Reload the page (`F5`). All records remain preserved via `data/simulation_history.json`.
  4. Test CSV log export or click `⟲ Clear Audit Log` to reset history.

---

## Known Bottlenecks & Architectural Limitations

As with any production machine learning system, several architectural trade-offs and bottlenecks have been identified:

1. **Static Contamination Assumption (`contamination=0.01`)**:
   * *Impact*: The pre-set $1\%$ anomaly threshold assumes a fixed anomaly ratio in live traffic. During seasonal consumption shifts (e.g., Black Friday or holiday sales), an uncalibrated static percentile can inflate false discovery rates without dynamic threshold adaptation.
2. **Stateless Scoring without Sequence Memory**:
   * *Impact*: The model evaluates each transaction vector $\mathbf{x} \in \mathbb{R}^d$ independently. It does not maintain stateful velocity counters (e.g., detecting 5 rapid swipes on the same card within 3 minutes across different cities).
3. **Euclidean Coordinate Approximation**:
   * *Impact*: Euclidean distance deltas $\Delta(\text{lat}, \text{long})$ provide lightning-fast vectorization but introduce map projection distortions at polar/extreme latitudes compared to strict orthodromic Great-Circle *Haversine* formulas.
4. **Local File I/O Concurrency for Audit Logs**:
   * *Impact*: Writing simulation logs to a local JSON file (`data/simulation_history.json`) is ideal for standalone deployments and demonstrations, but presents file lock/race conditions if scaled horizontally across multi-container Kubernetes pods without a centralized database.

---

## Engineering Roadmap & Future Improvements

To elevate FraudSentinel to enterprise banking scale:

1. **Hybrid Ensemble Architecture (Isolation Forest + LightGBM/XGBoost)**:
   * Couple the unsupervised isolation engine (for zero-day fraud pattern discovery) with a supervised gradient-boosted classifier tuned on verified chargeback histories.
2. **Real-Time Streaming Feature Store (Apache Kafka + Apache Flink + Redis)**:
   * Implement streaming sliding windows to calculate dynamic cardholder velocity metrics (e.g., transactions in the last 15 mins, expenditure vs. 30-day moving average).
3. **Exact TreeSHAP Explainability**:
   * Replace heuristic rule explanations with exact Shapley Additive Explanations (TreeSHAP) computed directly across Isolation Forest tree nodes for mathematical feature attribution.
4. **Microservice Decoupling & Containerized Production Deployment**:
   * Package the model into an asynchronous **FastAPI / gRPC** microservice deployed on **Docker + Kubernetes**, with real-time Data Drift monitoring via **Prometheus**, **Grafana**, and **Evidently AI**.

---

## Project Structure

```text
credit-fraud-detector/
├── .streamlit/
│   └── config.toml             # Streamlit server & CORS configuration
├── data/
│   └── simulation_history.json # Persistent local audit log
├── model/
│   ├── isolation_forest.pkl    # Serialized Isolation Forest model
│   ├── scaler.pkl              # Fitted StandardScaler pipeline
│   └── feature_names.pkl       # Feature schema alignment
├── app.py                      # Main Streamlit dashboard application
├── train.py                    # End-to-end dataset acquisition, training & evaluation
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Target Python runtime specification
├── README.md                   # Technical documentation (English)
└── README.pt-BR.md             # Technical documentation (Portuguese)
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git & Pip

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Marcelooll/credit-fraud-detector.git
cd credit-fraud-detector

# Create and activate virtual environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Training (Optional — Pre-trained weights included)

To retrain the Isolation Forest model directly from the raw Kaggle dataset:

```bash
# Ensure Kaggle API credentials (~/.kaggle/kaggle.json) are configured
python train.py
```

### 3. Running the Application

```bash
streamlit run app.py
```

Access the dashboard at your local host, or deploy it if you wish so.

---

## License

This project is released under the **MIT License**.
