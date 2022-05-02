import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


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

    # Fit SVM
    classifier = LinearSVC(C=1, random_state=0, dual=False)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_dev)

    # Save predictions and print accuracy
    np.savetxt(
        "predictions/SVM_dev_predictions.csv",
        y_pred,
        fmt="%i",
    )
    print(f"Accuracy: {metrics.accuracy_score(y_dev, y_pred)}")


if __name__ == "__main__":
    fit_predict()
