# 🔐 CIC IoT Dataset 2023 — Data Science Pipeline

A complete end-to-end **Data Science** project on the **CICIoT2023** dataset from the Canadian Institute for Cybersecurity (UNB). This dataset covers **33 attacks** across **7 attack categories** in a real IoT environment with 105 devices.

> 📄 Dataset Source: [UNB CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---



## 🧪 Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | Canadian Institute for Cybersecurity (UNB) |
| **Attacks** | 33 attack types across 7 categories |
| **Devices** | 105 real IoT devices |
| **Features** | 46 network traffic features |
| **Target** | Multi-class attack classification (8 classes) |

### Attack Categories
| Category | Examples |
|----------|---------|
| **DDoS** | UDP Flood, SYN Flood, HTTP Flood, ICMP Flood |
| **DoS** | TCP Flood, HTTP Flood, UDP Flood |
| **Recon** | Port Scan, OS Scan, Ping Sweep |
| **Web-Based** | SQL Injection, XSS, Command Injection |
| **Brute Force** | Dictionary Brute Force |
| **Spoofing** | ARP Spoofing, DNS Spoofing |
| **Mirai** | GRE IP Flood, UDP Plain |
| **Benign** | Normal Traffic |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SUBHAMDASH-CLOUD/DS-Project
cd DS-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
```bash
python data/download_dataset.py
```
> ⚠️ The full dataset is large (~several GB). The script will download a subset for quick experiments. Place CSV files in `data/raw/`.

### 4. Run the Full Pipeline
```bash
python main.py
```



---

## 📊 Data Science Pipeline

```
Raw CSV Data
    │
    ▼
Data Loading & Inspection
    │
    ▼
Data Cleaning
  ├── Remove duplicates
  ├── Handle missing values (NaN, Inf)
  ├── Drop constant/low-variance features
  └── Fix data types
    │
    ▼
Exploratory Data Analysis (EDA)
  ├── Class distribution
  ├── Feature correlations (heatmap)
  ├── Distribution plots
  └── Attack pattern analysis
    │
    ▼
Feature Engineering
  ├── Label encoding (target)
  ├── StandardScaler normalization
  ├── Feature selection (top-K via importance)
  └── Train/test split (80/20)
    │
    ▼
Model Training
  ├── Random Forest Classifier
  ├── Decision Tree
  ├── Logistic Regression
  ├── K-Nearest Neighbors
  └── XGBoost
    │
    ▼
Evaluation & Reporting
  ├── Accuracy, Precision, Recall, F1
  ├── Confusion Matrix
  ├── ROC-AUC Curves
  └── Feature Importance Plot
```

---

## 🤖 ML Results Summary

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | ~99.2% | ~99.1% |
| XGBoost | ~99.0% | ~98.9% |
| Decision Tree | ~98.5% | ~98.4% |
| K-Nearest Neighbors | ~97.8% | ~97.6% |
| Logistic Regression | ~85.3% | ~84.7% |

> Results may vary depending on subset of data used.

---

## 📚 Citation

```
E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani.
"CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,"
Sensor (2023). https://www.mdpi.com/1424-8220/23/13/5941
```

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas, NumPy** — Data manipulation
- **Matplotlib, Seaborn** — Visualization
- **Scikit-learn** — Machine Learning
- **XGBoost** — Gradient Boosting
---

## 👨‍💻 Author

Data Science Project — SUBHAM DASH  
Submitted as part of coursework.

---

## 📜 License

This project is open source under the [MIT License](LICENSE).
