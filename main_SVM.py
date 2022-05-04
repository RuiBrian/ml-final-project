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
        "datasets/processed/{}nt_train_separated.csv".format(num_NT), 
        header=None, index_col=False)
    X_train = X_train.to_numpy()
    X_train = OneHotEncoder().fit_transform(X_train).astype(int).toarray()
    y_train = np.load(
        "datasets/processed/{}nt_train_labels.npy".format(num_NT)).ravel()

    # Load dev data and labels
    X_dev = pd.read_csv(
        "datasets/processed/{}nt_dev_separated.csv".format(num_NT), 
        header=None, index_col=False)
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_dev = np.load(
        "datasets/processed/{}nt_dev_labels.npy".format(num_NT)).ravel()

    # Fit SVM
    classifier = LinearSVC(C=1, random_state=0, dual=False)
    model = CalibratedClassifierCV(classifier) 
    classifier.fit(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    y_softpred = model.predict_proba(X_dev)

    # Save predictions and print accuracy
    np.savetxt(
        "predictions/{}nt_SVM_dev_predictions.csv".format(num_NT),
        y_pred,
        fmt="%i",
        )
    np.savetxt(
        "predictions/{}nt_SVM_dev_softpredictions.csv".format(num_NT),
        y_softpred,
        fmt="%f",
        )
    print(f"Accuracy: {metrics.accuracy_score(y_dev, y_pred)}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        num_NT = sys.argv[1]
    elif len(sys.argv) == 1:
        num_NT = 80
    
    fit_predict(num_NT)
