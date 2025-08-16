import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

# ---------------------
# Utility / IO
# ---------------------

def load_default_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, "target"

def load_csv_dataset(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    return df, target

def preprocess_dataset(df, target_col):
    # Drop obvious ID / unnamed columns
    to_drop = [c for c in df.columns if c.lower().startswith("id") or c.lower().startswith("unnamed")]
    df = df.drop(columns=to_drop, errors="ignore")

    # Encode non-numeric target if needed
    if df[target_col].dtype == object or str(df[target_col].dtype).startswith("category"):
        df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

    # Impute numeric NaNs in features with mean
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    return df

# ---------------------
# Plotting helpers (matplotlib only; single-plot per figure)
# ---------------------

def plot_confusion_matrix(cm, classes, title, out_path):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(cax)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_accuracy_bar(acc_linear, acc_rbf, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(["Linear SVM", "RBF SVM"], [acc_linear, acc_rbf])
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_decision_boundary_2d(model, X_train, y_train, out_path):
    if X_train.shape[1] != 2:
        return
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", s=20)
    ax.set_title("Decision Boundary")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ---------------------
# Training / Evaluation
# ---------------------

def train_and_evaluate(df, target_col, features=None, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # Select features/target
    if features:
        X = df[features].values
    else:
        X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Linear SVM
    svm_linear = SVC(kernel="linear", probability=True, random_state=42)
    svm_linear.fit(X_train, y_train)
    y_pred_lin = svm_linear.predict(X_test)
    acc_linear = accuracy_score(y_test, y_pred_lin)

    # RBF SVM with GridSearch
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1]
    }
    svm_rbf = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1
    )
    svm_rbf.fit(X_train, y_train)
    best_rbf = svm_rbf.best_estimator_
    y_pred_rbf = best_rbf.predict(X_test)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)

    # Reports
    report_linear = classification_report(y_test, y_pred_lin, output_dict=True)
    report_rbf = classification_report(y_test, y_pred_rbf, output_dict=True)

    # Confusion matrices
    cm_linear = confusion_matrix(y_test, y_pred_lin)
    cm_rbf = confusion_matrix(y_test, y_pred_rbf)

    # Save metrics
    metrics = {
    "accuracy_linear": float(acc_linear),
    "accuracy_rbf": float(acc_rbf),
    "rbf_best_params": svm_rbf.best_params_,
    "classification_report_linear": report_linear,
    "classification_report_rbf": report_rbf
}

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plots
    plot_confusion_matrix(cm_linear, classes=[str(c) for c in np.unique(y)], title="Confusion Matrix (Linear SVM)",
                          out_path=os.path.join(out_dir, "confusion_matrix_linear.png"))
    plot_confusion_matrix(cm_rbf, classes=[str(c) for c in np.unique(y)], title="Confusion Matrix (RBF SVM)",
                          out_path=os.path.join(out_dir, "confusion_matrix_rbf.png"))
    plot_accuracy_bar(acc_linear, acc_rbf, os.path.join(out_dir, "accuracy_bar.png"))

    # Decision boundaries (only if 2 features)
    if X.shape[1] == 2:
        plot_decision_boundary_2d(svm_linear, X_train, y_train, os.path.join(out_dir, "decision_boundary_linear.png"))
        plot_decision_boundary_2d(best_rbf, X_train, y_train, os.path.join(out_dir, "decision_boundary_rbf.png"))

    print("Done. Metrics saved to outputs/metrics.json")

# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--features", nargs="+", help="Optional list of feature columns to use")
    args = parser.parse_args()

    if args.csv:
        if not args.target:
            raise ValueError("Please specify --target when using a custom CSV dataset.")
        df, target_col = load_csv_dataset(args.csv, args.target)
    else:
        df, target_col = load_default_dataset()

    df = preprocess_dataset(df, target_col)
    train_and_evaluate(df, target_col, args.features)

if __name__ == "__main__":
    main()