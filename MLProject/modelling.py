"""
modelling.py
=============
Machine Learning pipeline tanpa hyperparameter tuning, dengan MLflow manual logging.
Dataset : Enhanced Superstore Sales — hasil preprocessing dari folder
          preprocessing/namadataset_preprocessing/

Cara menjalankan
----------------
1. Install dependensi:
       pip install mlflow scikit-learn pandas numpy matplotlib seaborn dagshub

2a. Local:
       python modelling.py
       mlflow ui --port 5000
       Buka http://localhost:5000

2b. DagsHub:
       python modelling.py --dagshub --repo-owner <username> --repo-name <repo>
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)

warnings.filterwarnings("ignore")


# ── Konfigurasi ────────────────────────────────────────────────────────────────
PREPROC_DIR  = "preprocessing/namadataset_preprocessing"
ARTIFACT_DIR = "artifacts"
EXPERIMENT   = "Superstore_Profit_Classification_Base"
RANDOM_STATE = 42

MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE
    ),
}

MODEL_PARAMS = {
    "RandomForest":      {"n_estimators": 100, "max_depth": "None", "min_samples_split": 2},
    "GradientBoosting":  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    "LogisticRegression":{"C": 1.0, "solver": "lbfgs", "penalty": "l2"},
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def log(msg): print(f"[modelling] {msg}")


def load_data(preproc_dir: str):
    X_train = pd.read_csv(f"{preproc_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{preproc_dir}/X_test.csv")
    y_train = pd.read_csv(f"{preproc_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{preproc_dir}/y_test.csv").squeeze()
    log(f"Data dimuat — train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    return m


# ── Artefak ───────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, model_name, out_dir) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rugi", "Untung"], yticklabels=["Rugi", "Untung"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def save_classification_report(y_true, y_pred, model_name, out_dir) -> str:
    report = classification_report(y_true, y_pred, target_names=["Rugi", "Untung"])
    path = os.path.join(out_dir, f"classification_report_{model_name}.txt")
    with open(path, "w") as f:
        f.write(f"Classification Report — {model_name}\n{'='*50}\n{report}")
    return path


def save_feature_importance(model, feature_names, model_name, out_dir):
    if not hasattr(model, "feature_importances_"):
        return None
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance (Top 20) — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(out_dir, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def save_roc_curve(y_true, y_prob, model_name, out_dir):
    if y_prob is None:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}", color="teal")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def save_metrics_comparison(all_metrics: dict, out_dir: str) -> str:
    df_m = pd.DataFrame(all_metrics).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    df_m.plot(kind="bar", ax=ax, edgecolor="white", width=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_title("Perbandingan Metrik Antar Model (Base)")
    ax.set_ylabel("Score")
    ax.set_xticklabels(df_m.index, rotation=15)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_comparison_base.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


# ── Training + MLflow logging ─────────────────────────────────────────────────
def train_and_log(model_name, model, params,
                  X_train, X_test, y_train, y_test,
                  artifact_dir) -> dict:
    log(f"--- Melatih: {model_name} ---")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_test, y_pred, y_prob)

    with mlflow.start_run(run_name=model_name):
        # Tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset",    "Enhanced Superstore Sales")
        mlflow.set_tag("task",       "binary_classification")
        mlflow.set_tag("tuning",     "none")

        # Parameters
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size",    0.2)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Metrics
        for name, val in metrics.items():
            mlflow.log_metric(name, round(val, 6))

        # Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:5],
        )

        # Artefak tambahan
        cm_path = save_confusion_matrix(y_test, y_pred, model_name, artifact_dir)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        cr_path = save_classification_report(y_test, y_pred, model_name, artifact_dir)
        mlflow.log_artifact(cr_path, artifact_path="reports")

        fi_path = save_feature_importance(model, X_train.columns.tolist(), model_name, artifact_dir)
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="plots")

        roc_path = save_roc_curve(y_test, y_prob, model_name, artifact_dir)
        if roc_path:
            mlflow.log_artifact(roc_path, artifact_path="plots")

        log(f"  Run ID: {mlflow.active_run().info.run_id}")

    log(f"  Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}  "
        f"AUC={metrics.get('roc_auc', 'N/A')}")
    return metrics


# ── Pipeline utama ─────────────────────────────────────────────────────────────
def run(preproc_dir, use_dagshub, repo_owner, repo_name):
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")

    if use_dagshub:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        log(f"DagsHub aktif: https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    elif env_uri and env_uri != "mlruns":
        # Pakai URI dari environment variable (CI/DagsHub via secrets)
        mlflow.set_tracking_uri(env_uri)
        log(f"MLflow Tracking URI dari env: {env_uri}")
    else:
        mlflow.set_tracking_uri("mlruns")
        log("Local MLflow — jalankan `mlflow ui` untuk melihat hasil.")

    mlflow.set_experiment(EXPERIMENT)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(preproc_dir)

    all_metrics = {}
    for model_name, model in MODELS.items():
        all_metrics[model_name] = train_and_log(
            model_name, model, MODEL_PARAMS[model_name],
            X_train, X_test, y_train, y_test, ARTIFACT_DIR,
        )

    comp_path = save_metrics_comparison(all_metrics, ARTIFACT_DIR)
    with mlflow.start_run(run_name="Model_Comparison_Base"):
        mlflow.log_artifact(comp_path, artifact_path="plots")
        for mname, mvals in all_metrics.items():
            for k, v in mvals.items():
                mlflow.log_metric(f"{mname}_{k}", round(v, 6))

    print("\n" + "=" * 55)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 55)
    for mname, mvals in all_metrics.items():
        auc = f"{mvals.get('roc_auc', float('nan')):.4f}"
        print(f"{mname:<22} {mvals['accuracy']:>10.4f} {mvals['f1_score']:>10.4f} {auc:>10}")
    print("=" * 55)
    best = max(all_metrics, key=lambda m: all_metrics[m]["f1_score"])
    log(f"Model terbaik: {best}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelling base tanpa tuning — MLflow manual logging")
    parser.add_argument("--preproc-dir", default=PREPROC_DIR)
    parser.add_argument("--dagshub", action="store_true")
    parser.add_argument("--repo-owner", default="")
    parser.add_argument("--repo-name",  default="")
    args = parser.parse_args()

    if args.dagshub and not (args.repo_owner and args.repo_name):
        parser.error("--dagshub membutuhkan --repo-owner dan --repo-name")

    run(args.preproc_dir, args.dagshub, args.repo_owner, args.repo_name)