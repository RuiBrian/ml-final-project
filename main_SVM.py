import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import learning_curve


def fit_predict(flanking_seq, dataset):
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
        f"datasets/processed/{flanking_seq}nt_{dataset}_separated.csv",
        header=None,
        index_col=False,
    )
    X_dev = X_dev.to_numpy()
    X_dev = OneHotEncoder().fit_transform(X_dev).astype(int).toarray()
    y_dev = np.load(f"datasets/processed/{flanking_seq}nt_{dataset}_labels.npy").ravel()

    # Fit SVM
    clf = LinearSVC(C=1, random_state=0, dual=False)
    calibrated_clf = CalibratedClassifierCV(clf)

    clf.fit(X_train, y_train)
    calibrated_clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev)
    y_softpred = calibrated_clf.predict_proba(X_dev)

    # Plot learning curve
    plot_learning_curve(clf, f"SVM Learning Curves {flanking_seq}nt", X_train, y_train)
    plt.savefig(f"logs/{flanking_seq}nt_traindev_SVM.png")
    # plt.show()

    # Save predictions and print accuracy
    np.savetxt(
        f"predictions/{flanking_seq}nt_SVM_{dataset}_predictions.csv",
        y_pred,
        fmt="%i",
    )
    np.savetxt(
        f"predictions/{flanking_seq}nt_SVM_{dataset}_softpredictions.csv",
        y_softpred,
        fmt="%f",
    )
    print(f"Accuracy: {metrics.accuracy_score(y_dev, y_pred)}")


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
):

    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="b", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="r", label="Cross-validation score"
    )
    plt.legend(loc="best")

    return plt


if __name__ == "__main__":
    flanking_seq = 80
    datatype = "dev"

    if len(sys.argv) >= 2:
        flanking_seq = int(sys.argv[1])
        if len(sys.argv) == 3:
            datatype = sys.argv[2]

    fit_predict(flanking_seq, datatype)
