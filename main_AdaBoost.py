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
    plot_learning_curve(clf, f"AdaBoost Learning Curves {flanking_seq}nt", X_train, y_train)
    plt.figure(figsize=(15, 5))
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


def plot_learning_curve(estimator, title, X, y,
    axes=None,
    ylim=None,
    cv=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    ):
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, return_times=True,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, "o-")
    # axes[1].fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    # )
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # axes[2].grid()
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    # axes[2].fill_between(
    #     fit_time_sorted,
    #     test_scores_mean_sorted - test_scores_std_sorted,
    #     test_scores_mean_sorted + test_scores_std_sorted,
    #     alpha=0.1,
    # )
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt


if __name__ == "__main__":
    flanking_seq = 80
    datatype = "dev"

    if len(sys.argv) >= 2:
        flanking_seq = int(sys.argv[1])
        if len(sys.argv) == 3:
            datatype = sys.argv[2]

    fit_predict(flanking_seq, datatype)
