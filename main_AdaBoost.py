import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier


def fit_predict():
    # Load training data and labels
    X_train = pd.read_csv(
        "datasets/processed/train_separated.csv", header=None, index_col=False
    )
    X_train = X_train.to_numpy()
    X_train = OneHotEncoder().fit_transform(X_train).astype(int).toarray()
    y_train = np.load("datasets/processed/train_labels.npy").ravel()

    # Load dev data and labels
    X_dev = pd.read_csv(
        "datasets/processed/dev_separated.csv", header=None, index_col=False
    )
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_dev = np.load("datasets/processed/dev_labels.npy").ravel()

    # Fit AdaBoost classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Save predictions and print accuracy
    np.savetxt(
        "predictions/AdaBoost_dev_predictions.csv",
        clf.predict(X_dev).astype(int),
        fmt="%i",
    )
    np.savetxt(
        "predictions/AdaBoost_dev_softpredictions.csv",
        clf.predict_proba(X_dev).astype(float),
        fmt="%f",
    )
    print(f"Accuracy: {clf.score(X_dev, y_dev)}")


if __name__ == "__main__":
    fit_predict()