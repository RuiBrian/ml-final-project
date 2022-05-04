import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier


def fit_predict(num_NT):
    # Load training data and labels
    X_train = pd.read_csv(
        "datasets/processed/{}nt_train_separated.csv".format(num_NT), header=None, index_col=False
    )
    X_train = X_train.to_numpy()
    X_train = OneHotEncoder().fit_transform(X_train).astype(int).toarray()
    y_train = np.load("datasets/processed/{}nt_train_labels.npy".format(num_NT)).ravel()

    # Load dev data and labels
    X_dev = pd.read_csv(
        "datasets/processed/{}nt_dev_separated.csv".format(num_NT), header=None, index_col=False
    )
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_dev = np.load("datasets/processed/{}nt_dev_labels.npy".format(num_NT)).ravel()

    # Fit AdaBoost classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Save predictions and print accuracy
    np.savetxt(
        "predictions/{}nt_AdaBoost_dev_predictions.csv".format(num_NT),
        clf.predict(X_dev).astype(int),
        fmt="%i",
    )
    np.savetxt(
        "predictions/{}nt_AdaBoost_dev_softpredictions.csv".format(num_NT),
        clf.predict_proba(X_dev).astype(float),
        fmt="%f",
    )
    print(f"Accuracy: {clf.score(X_dev, y_dev)}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        num_NT = sys.argv[1]
    elif len(sys.argv) == 1:
        num_NT = 80
    
    fit_predict(num_NT)
