import numpy as np
import pandas as pd
import sys

from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV


def fit_predict(num_NT):
    # Load training data and labels
    X_train = pd.read_csv(
        f"datasets/processed/{num_NT}nt_train_separated.csv",
        header=None,
        index_col=False,
    )
    X_train = X_train.to_numpy()
    X_train = OneHotEncoder().fit_transform(X_train).astype(int).toarray()
    y_train = np.load(f"datasets/processed/{num_NT}nt_train_labels.npy").ravel()

    # Load dev data and labels
    X_dev = pd.read_csv(
        f"datasets/processed/{num_NT}nt_dev_separated.csv",
        header=None,
        index_col=False,
    )
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_dev = np.load(f"datasets/processed/{num_NT}nt_dev_labels.npy").ravel()

    # Fit SVM
    clf = LinearSVC(C=1, random_state=0, dual=False)
    calibrated_clf = CalibratedClassifierCV(clf)

    clf.fit(X_train, y_train)
    calibrated_clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev)
    y_softpred = calibrated_clf.predict_proba(X_dev)

    # Save predictions and print accuracy
    np.savetxt(
        f"predictions/{num_NT}nt_SVM_dev_predictions.csv",
        y_pred,
        fmt="%i",
    )
    np.savetxt(
        f"predictions/{num_NT}nt_SVM_dev_softpredictions.csv",
        y_softpred,
        fmt="%f",
    )
    print(f"Accuracy: {metrics.accuracy_score(y_dev, y_pred)}")


if __name__ == "__main__":
    num_NT = 80

    if len(sys.argv) == 2:
        num_NT = int(sys.argv[1])

    fit_predict(num_NT)
