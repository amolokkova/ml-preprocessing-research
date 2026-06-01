from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


POSITIVE_CLASS = ">50K"
TARGET_COLUMN = "class"
BREAST_CANCER_TARGET_COLUMN = "target"


def load_adult_data(path: str) -> pd.DataFrame:
    """Load Adult Census data and normalize missing values and numeric columns."""
    df = pd.read_csv(path, na_values="?", skipinitialspace=True)

    numeric_columns = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def load_breast_cancer_data() -> pd.DataFrame:
    """Load Breast Cancer Wisconsin data with readable target labels."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[BREAST_CANCER_TARGET_COLUMN] = pd.Series(data.target).map(
        {
            0: "malignant",
            1: "benign",
        }
    )
    return df


def split_breast_cancer_features_target(df: pd.DataFrame):
    X = df.drop(columns=[BREAST_CANCER_TARGET_COLUMN])
    y = df[BREAST_CANCER_TARGET_COLUMN]
    return X, y


def get_feature_groups(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features


def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def calculate_binary_metrics(y_true, y_pred, y_proba, positive_label):
    y_true_binary = pd.Series(y_true).eq(positive_label).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred, pos_label=positive_label),
        "ROC-AUC": roc_auc_score(y_true_binary, y_proba),
    }


def save_results(df: pd.DataFrame, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_figure(path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
