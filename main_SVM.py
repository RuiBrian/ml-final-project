import numpy as np
import pandas as pd

from sklearn.svm import SVC
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

    # Load test data and labels
    X_test = pd.read_csv(
        "datasets/processed/test_separated.csv", header=None, index_col=False
    )
    X_test = X_test.to_numpy()
    X_test = OneHotEncoder().fit_transform(X_test).astype(int).toarray()
    y_test = np.load("datasets/processed/test_labels.npy").ravel()

    classifier = SVC(kernel="linear", C=1, decision_function_shape="ovo")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


if __name__ == "__main__":
    fit_predict()
