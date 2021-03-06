from distutils.log import ERROR
from typing import Dict
import numpy as np
import sys
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import csv


def simple_nn_accuracy(file):
    data = np.loadtxt(file, dtype=int, delimiter=",", skiprows=1)
    true = data[:, 1]
    pred = data[:, 2]
    comparison = true == pred
    accuracy = ((np.sum(comparison)) / comparison.size).astype(np.float64)
    acc = dict()
    classes = np.unique(true)
    correct = 0
    # for i in range(len(true)):
    #     if true[i] == pred[i]:
    #         correct+=1
    # print(correct/len(true))
    # print(len(preds))
    for i in classes:
        temp_l = true[np.where(true == i)]
        temp_preds = pred[np.where(true == i)]
        temp_comp = temp_l == temp_preds
        acc[i] = np.sum(temp_comp) / temp_comp.size
    return accuracy, acc


def simple_sa_accuracy(file):
    data = np.loadtxt(file, dtype=int, delimiter=",", skiprows=1)
    true = data[:, 0]
    pred = data[:, 1]
    comparison = true == pred
    accuracy = ((np.sum(comparison)) / comparison.size).astype(np.float64)
    acc = dict()
    classes = np.unique(true)
    correct = 0
    # for i in range(len(true)):
    #     if true[i] == pred[i]:
    #         correct+=1
    # print(correct/len(true))
    # print(len(preds))
    for i in classes:
        temp_l = true[np.where(true == i)]
        temp_preds = pred[np.where(true == i)]
        temp_comp = temp_l == temp_preds
        acc[i] = np.sum(temp_comp) / temp_comp.size
    return accuracy, acc


def simple_accuracy(truefile, predfile):
    labels = np.load(truefile).squeeze()
    preds = np.loadtxt(predfile)
    if labels.shape != preds.shape:
        print("labels len and pred len doesn't match")
        return
    comparison = labels == preds
    classes = np.unique(labels)
    acc = dict()
    # print(len(preds))
    for i in classes:
        temp_l = labels[np.where(labels == i)]
        temp_preds = preds[np.where(labels == i)]
        temp_comp = temp_l == temp_preds
        acc[i] = np.sum(temp_comp) / temp_comp.size

    # incorrect = np.where(comparison=False)
    # print(incorrect)
    # accuracy = len(incorrect[0])/len(labels)
    accuracy = ((np.sum(comparison)) / comparison.size).astype(np.float64)
    return accuracy, acc


def nn_pr_auc(file):
    """
    PR-AUC is the average area under the precision-recall curve of each class.
    Binarize predictions and labels then calculate precision-recall
    curve for each class
    """
    data = np.loadtxt(file, dtype=int, delimiter=",", skiprows=1)
    classes = [0, 1, 2]
    true = data[:, 1]
    bintrue = label_binarize(true, classes=classes)
    pred = data[:, 2]
    binpred = label_binarize(pred, classes=classes)
    # for i in range(len(pred)):
    #     if pred[i] == 3:
    #         if not true[i] == 0:
    #             comparison[i] = True
    precision = dict()
    recall = dict()
    threshold = dict()
    accuracies = dict()
    accuracies = []
    cm = confusion_matrix(true, pred, labels=classes)
    plt.axis()
    for i in range(len(classes)):
        precision[i], recall[i], threshold[i] = precision_recall_curve(
            bintrue[:, i], binpred[:, i]
        )
        plt.plot(recall[i], precision[i], label=f"class {i}")
        accuracies.append(auc(recall[i], precision[i]))
        # accuracies.append(average_precision_score(true, pred))
    plt.legend()
    accuracy = np.mean(accuracies) * 100
    figname = os.path.splitext(file)[0]
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(figname + " " + str(accuracy))
    plt.savefig(figname + ".png")
    plt.show()
    # # print(accuracies)
    # print(threshold)
    # print(recall)
    # print(precision)
    # precision, recall, thresholds = precision_recall_curve(true, pred)
    # accuracy = auc(precision, recall)
    return accuracy, cm


def pr_auc(truefile, probfile, predfile):
    """
    PR-AUC is the area under the precision-recall curve.
    truefile = labels of data (Nx1)
    probfile = probabality sample belongs to each class (Nx3)
    """
    classes = [0, 1, 2]
    labels = np.load(truefile, allow_pickle=True)
    preds = np.loadtxt(predfile, dtype=int)
    binpreds = label_binarize(preds, classes=classes)
    probs = np.loadtxt(probfile, dtype=float)
    # probpred = np.argmax(probs,1)
    # print(len(probpred))
    # print(len(np.where(probpred != preds)[0]))
    # print(np.where(probpred != preds))
    # convert labels into Nx3 binary matrix
    binlabels = label_binarize(labels, classes=classes)
    precision = dict()
    recall = dict()
    threshold = dict()
    accuracies = []
    plt.axis()
    # *** double check softpredictions match predictions - they do!
    cm = confusion_matrix(labels, preds, labels=classes)

    for i in range(len(classes)):
        temp = binlabels[:, i]
        precision[i], recall[i], threshold[i] = precision_recall_curve(
            temp, probs[:, i]
        )
        accuracies.append(auc(recall[i], precision[i]))
        plt.plot(recall[i], precision[i], label=f"class {i}")
        no_skill = len(temp[temp == 1]) / len(temp)  # *** check for bug here?
        plt.plot(
            [0, 1],
            [no_skill, no_skill],
            linestyle="--",
            label=f"Baseline (Positive Proportion of {i})",
        )

    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    figname = os.path.splitext(probfile)[0]
    plt.title(figname)
    plt.savefig(figname + ".png")
    plt.show()

    accuracy = np.mean(accuracies)
    # print(f"length of PR should be {len(classes)} and is {len(precision)}."+
    #     f"Num thresholds for each class is {len(threshold[0])}. "
    #     f"Num precision for each class is {len(precision[0])}")

    return accuracy, cm


def nn_top_k_accuracy(file):
    # TODO
    """
    Top-k accuracy is the fraction of correctly predicted splice sites
    at the threshold where the number of predicted sites is equal to
    the actual number of sites present.
    """
    raise NotImplementedError()


def top_k_accuracy(truefile, predfile):
    # TODO
    """
    Top-k accuracy is the fraction of correctly predicted splice sites
    at the threshold where the number of predicted sites is equal to
    the actual number of sites present.
    """
    raise NotImplementedError()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("***Please provide model being assessed and flanking length***")
        sys.exit()

    MODEL = sys.argv[1]
    if MODEL == "nnsplice":
        flanking = ["80", "400"]
        for f in flanking:
            outputfile = f"output/{f}nt_merged_nn_preds1.csv"
            print(f"{outputfile} accuracy is {simple_nn_accuracy(outputfile)}")
            print(f"and PR-AUC = {nn_pr_auc(outputfile)}")

    elif MODEL == "ourmodel":

        if len(sys.argv) == 3:
            flanking = sys.argv[2]
        else:
            raise Exception("Flanking sequence must be 80 or 400")

        truef = f"datasets/processed/{flanking}nt_dev_labels.npy"
        classifiers = ["AdaBoost", "CNN", "SVM"]
        for c in classifiers:
            predf = f"predictions/{flanking}nt_{c}_dev_predictions.csv"
            predf2 = f"predictions/{flanking}nt_{c}_dev_softpredictions.csv"
            print(f"{predf} simple accuracy={simple_accuracy(truef,predf)}")
            print(f"{predf2} pr-auc={pr_auc(truef,predf2,predf)}")
    elif MODEL == "spliceai":
        if len(sys.argv) == 3:
            flanking = sys.argv[2]
        else:
            raise Exception("Flanking sequence must be 80 or 400")
        print(f"acc: {simple_sa_accuracy(f'output/{flanking}nt_spliceai_preds_0.csv')}")
