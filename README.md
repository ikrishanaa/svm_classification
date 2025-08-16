# Support Vector Machine (SVM) Classification Project

## Overview
This project implements Support Vector Machine (SVM) classifiers with both **Linear** and **RBF** kernels. It includes preprocessing, model training, hyperparameter tuning, evaluation, and optional 2D decision boundary visualization.

If no dataset is provided, it defaults to the **Breast Cancer Wisconsin** dataset from scikit-learn.

## Features
- Load dataset from CSV (`--csv`) and specify target (`--target`), or use the default dataset.
- Preprocessing: handle missing values, drop obvious ID/unnamed columns, and encode non-numeric targets.
- Feature scaling with `StandardScaler`.
- Train **Linear SVM** and **RBF SVM**.
- Hyperparameter tuning for RBF SVM using `GridSearchCV` over `C` and `gamma`.
- Cross-validated metrics, confusion matrices, and classification reports saved to disk.
- Decision boundary plots when exactly **two features** are used.

## Folder Structure
```
svm_classification_project/
│
├── data/                 
├── outputs/              
├── src/                  
│   └── svm_model.py
├── README.md
└── requirements.txt
```

## Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1) Default dataset (Breast Cancer)
```bash
python src/svm_model.py
```

### 2) Custom dataset
```bash
python src/svm_model.py --csv data/your_dataset.csv --target target_column_name
```

### 3) Custom dataset with exactly two features (enables decision boundaries)
```bash
python src/svm_model.py --csv data/your_dataset.csv --target target_column_name --features feature1 feature2
```

## Outputs
- `outputs/metrics.json` – accuracy and best parameters
- `outputs/confusion_matrix_linear.png`
- `outputs/confusion_matrix_rbf.png`
- `outputs/decision_boundary_linear.png` (if 2D)
- `outputs/decision_boundary_rbf.png` (if 2D)
- `outputs/accuracy_bar.png`

## Notes
- For categorical targets (e.g., "M"/"B"), the script automatically encodes them.
- Missing numeric values are imputed with the column mean.
- If you do **not** pass `--target` with a CSV, the script will prompt you to specify it.
