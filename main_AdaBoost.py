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

    # Load test data and labels
    X_dev = pd.read_csv(
        "datasets/processed/test_separated.csv", header=None, index_col=False
    )
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_test = np.load("datasets/processed/test_labels.npy").ravel()

    # Fit AdaBoost classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Save predictions and print accuracy
    np.savetxt(
        "predictions/AdaBoost_dev_predictions.csv",
        clf.predict(X_dev).astype(int),
        fmt="%i",
    )
    print(f"Accuracy: {clf.score(X_dev, y_test)}")


if __name__ == "__main__":
    fit_predict()
