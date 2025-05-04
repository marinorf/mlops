# 🤖 ML Ops Workflow for Breast Cancer Wisconsin

This repository implements a full ML workflow for the Breast Cancer Wisconsin dataset, developed as part of the ML Operations course at EWHA University. All steps—from data ingestion through model training and evaluation—are managed through a Makefile.

---

## ⚙️ Workflow Overview

1. **Data Management**  
2. **Preprocessing**  
3. **Model Training**  
4. **Model Evaluation**  
5. **Dependency Management**

---

### 1. Data Management

#### Download Data  
Fetch the Breast Cancer Wisconsin dataset from Kaggle and save it as `data/raw.csv`.

<blockquote>
<pre>
make download-data
</pre>
</blockquote>

> **Prerequisite:** You must have a valid `kaggle.json` in `~/.kaggle/` (or set `KAGGLE_CONFIG_DIR`) for Kaggle API authentication.

---

### 2. Preprocessing

#### Preprocess Raw Data  
Clean the raw CSV, engineer features, and split into train/test sets:
- `data/X_train.csv`
- `data/y_train.csv`
- `data/X_test.csv`
- `data/y_test.csv`

<blockquote>
<pre>
make preprocess-data
</pre>
</blockquote>

---

### 3. Model Training

#### Train CatBoost Model  
Train a CatBoost classifier on the processed training data and save the model artifact under `models/`.

<blockquote>
<pre>
make train-catboost
</pre>
</blockquote>

#### Train MLJAR AutoML Model  
Launch an AutoML run with MLJAR‑Supervised, storing results in `models/AutoML_results/`.

<blockquote>
<pre>
make train-MLJar
</pre>
</blockquote>

---

### 4. Model Evaluation

#### Evaluate All Models  
Compute accuracy, precision, recall, F1, confusion matrices, and output predictions under `evaluate/`.

<blockquote>
<pre>
make evaluate-models
</pre>
</blockquote>

---

### 5. Dependency Management

#### Install Python Packages  
Install all required libraries as pinned in `requirements.txt`.

<blockquote>
<pre>
make install
</pre>
</blockquote>

---

## 📂 Repository Structure
```
.
├── Makefile                     # Automates the workflow
├── requirements.txt             # Python dependencies
├── data/                        # Raw & preprocessed datasets
│   ├── raw.csv
│   ├── processed/
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_test.csv
│       └── y_test.csv
├── models/                      # Model python files and saved models
│   ├── best_catboost_model.cbm
│   ├── catboost_model.py
│   ├── MLJar_model.py
│   └── AutoML_results/         
├── scripts/                     # Pipeline scripts
│   └── preprocess.py            # Data cleaning & train/test seperation
├── evaluate/                    # Model evaluation scripts
│   ├── evaluate_catboost.py
│   └── evaluate_MLJar.py
```

### 📝 Notes
Testing - tests of functions are not yet implemented.

Kaggle Authentication - Ensure your Kaggle API token (kaggle.json) is in ~/.kaggle/.