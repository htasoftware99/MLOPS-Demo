import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

# Dagshub integration
dagshub.init(repo_owner='hasan.asus1999', repo_name='MLOPS-Demo', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_classification_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred, average='weighted')
    rec = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return acc, prec, rec, f1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Veri seti: Iris
    csv_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    try:
        data = pd.read_csv(csv_url)
    except Exception as e:
        logger.exception("Unable to download CSV. Error: %s", e)

    # Özellikleri ve hedefi ayır
    X = data.drop("species", axis=1)
    y = data["species"]

    # Eğitim/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Hiperparametreler
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc, prec, rec, f1 = eval_classification_metrics(y_test, y_pred)

        print(f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}):")
        print(f"  Accuracy: {acc}")
        print(f"  Precision: {prec}")
        print(f"  Recall: {rec}")
        print(f"  F1 Score: {f1}")

        # MLflow loglamaları
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Dagshub URI
        remote_server_uri = "https://dagshub.com/hasan.asus1999/MLOPS-Demo.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(clf, "model", registered_model_name="RandomForestIrisModel")
        else:
            mlflow.sklearn.log_model(clf, "model")
