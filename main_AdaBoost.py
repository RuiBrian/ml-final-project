import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve


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

    # Plot learning curve
    plot_learning_curve(
        clf, f"AdaBoost Learning Curves {flanking_seq}nt", X_train, y_train
    )
    plt.savefig(f"logs/{flanking_seq}nt_traindev_AdaBoost.png")
    # plt.show()

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
