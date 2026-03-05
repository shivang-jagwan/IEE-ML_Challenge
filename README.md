# IEEE ML Challenge — Fault Detection

Binary classification on sensor data to detect **faulty vs normal** devices.  
Built for the **ML Challenge by IEEE SB, GEHU** (online qualifiers).

---

## Problem Statement

Given 47 numerical features (`F01`–`F47`) captured by an embedded monitoring system, predict whether a device is operating normally or experiencing a fault condition.

| Class | Label   |
|-------|---------|
| 0     | Normal  |
| 1     | Faulty  |

---

## Dataset

| File       | Rows   | Columns | Description                          |
|------------|--------|---------|--------------------------------------|
| TRAIN.csv  | 43,776 | 48      | 47 features + Class label            |
| TEST.csv   | 10,944 | 48      | 47 features + predicted Class        |
| FINAL.csv  | 10,944 | 2       | ID + CLASS (submission format)       |

- **No missing values** in either file
- **738 duplicate rows** in training data (dropped during preprocessing)
- Class distribution: ~60% Normal / ~40% Faulty

---

## Approach

### 1. Preprocessing
- Removed duplicate rows from training data
- No imputation needed (zero missing values)

### 2. Feature Engineering (47 → 72 features)
- **Row-wise statistics**: mean, std, min, max, range across all 47 features
- **Log transforms**: `log1p` on heavy-tailed features (F30–F38)
- **Interaction terms**: products and ratios of correlated feature pairs (e.g., F01×F09, F19÷F21)
- **Group sums**: summed features in logical groups (F01–F09, F10–F18, F19–F29, F30–F38, F39–F47)

### 3. Models
Ensemble of 3 gradient boosting models with **soft voting**:

| Model      | Type     | Trees | Depth | Learning Rate |
|------------|----------|-------|-------|---------------|
| XGBoost    | XGBClassifier (hist) | 300 | 5 | 0.05 |
| LightGBM-1 | LGBMClassifier | 300 | 5 | 0.05 |
| LightGBM-2 | LGBMClassifier | 250 | 6 | 0.08 |

All models use **strong regularization** to prevent overfitting:
- `reg_alpha` (L1): 0.5–1.0
- `reg_lambda` (L2): 3.0–5.0
- `subsample`: 0.7
- `colsample_bytree`: 0.5–0.6
- Class imbalance handled via `scale_pos_weight` (XGBoost) and `is_unbalance` (LightGBM)

### 4. Validation
- **3-fold Stratified CV** for quick accuracy check
- **80/20 holdout split** with full metrics: Accuracy, F1, Precision, Recall, ROC-AUC, MCC, Confusion Matrix
- **Sanity check**: predicting on training data itself to verify model consistency

---

## Output

**FINAL.csv** — submission file in the required format:

```
ID,CLASS
1,0
2,1
3,0
...
```

---

## How to Run

### Requirements
```
pandas
numpy
scikit-learn
xgboost
lightgbm
```

### Run on Google Colab
1. Upload `TRAIN.csv` and `TEST.csv` to Colab
2. Open `solution.ipynb`
3. Run all cells (takes ~3–5 minutes)
4. Download `FINAL.csv` for submission

### Run Locally
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
jupyter notebook solution.ipynb
```

---

## Repository Structure

```
├── README.md          # this file
├── solution.ipynb     # full pipeline notebook
├── TRAIN.csv          # training data (47 features + Class)
├── TEST.csv           # test data with predicted Class column
├── FINAL.csv          # submission file (ID + CLASS)
└── readme.txt         # original challenge description
```

---

## Tech Stack

- Python 3.10+
- XGBoost (histogram-based)
- LightGBM
- scikit-learn (VotingClassifier, metrics, CV)
- pandas / numpy
