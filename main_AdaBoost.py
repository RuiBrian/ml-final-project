import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier


def fit_predict(flanking_seq, datatype="dev"):
    # Load training data and labels
    X_train = pd.read_csv(
        f"datasets/processed/{flanking_seq}nt_train_separated.csv",
        header=None,
        index_col=False,
    )
    X_train = X_train.to_numpy()
    X_train = OneHotEncoder().fit_transform(X_train).astype(int).toarray()
    y_train = np.load(f"datasets/processed/{flanking_seq}nt_train_labels.npy").ravel()

    # Load dev data and labels
    X_dev = pd.read_csv(
        f"datasets/processed/{flanking_seq}nt_{datatype}_separated.csv",
        header=None,
        index_col=False,
    )
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_dev = np.load(
        f"datasets/processed/{flanking_seq}nt_{datatype}_labels.npy"
    ).ravel()

    # Fit AdaBoost classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Save predictions and print accuracy
    np.savetxt(
        f"predictions/{flanking_seq}nt_AdaBoost_{datatype}_predictions.csv",
        clf.predict(X_dev).astype(int),
        fmt="%i",
    )
    np.savetxt(
        f"predictions/{flanking_seq}nt_AdaBoost_{datatype}_softpredictions.csv",
        clf.predict_proba(X_dev).astype(float),
        fmt="%f",
    )
    print(f"Accuracy: {clf.score(X_dev, y_dev)}")


if __name__ == "__main__":
    flanking_seq = 80
    datatype = "dev"

    if len(sys.argv) >= 2:
        flanking_seq = int(sys.argv[1])
        if len(sys.argv) == 3:
            datatype = sys.argv[2]

    fit_predict(flanking_seq, datatype)
