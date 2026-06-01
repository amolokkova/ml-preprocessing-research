from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


POSITIVE_CLASS = ">50K"
TARGET_COLUMN = "class"


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


def save_results(df: pd.DataFrame, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_figure(path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
