from distutils.log import ERROR
from typing import Dict
from xml.etree.ElementPath import prepare_descendant
import numpy as np
import sys
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from torchmetrics import PrecisionRecallCurve, AveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from torchmetrics.functional import auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os

def simple_nn_accuracy(file):
    data = np.loadtxt(file, dtype=int, delimiter=",", skiprows=1)
    true = data[:, 1]
    pred = data[:, 2]
    comparison = true == pred
    # incorrect = np.where(true!=pred)
    # print(len(incorrect[0]))
    # print(len(true))
    # accuracy = len(incorrect[0])/len(true)
    accuracy = ((np.sum(comparison)) / comparison.size).astype(np.float64)
    return accuracy


def simple_accuracy(truefile, predfile):
    labels = np.load(truefile).squeeze()
    preds = np.loadtxt(predfile)
    if labels.shape != preds.shape:
        print("labels len and pred len doesn't match")
        return
    comparison = (labels == preds)
    # incorrect = np.where(comparison=False)
    # print(incorrect)
    # accuracy = len(incorrect[0])/len(labels)
    accuracy = ((np.sum(comparison)) / comparison.size).astype(np.float64)
    return accuracy


def nn_pr_auc(file):
    """
    PR-AUC is the area under the precision-recall curve.
    Binarize predictions and labels then calculate precision-recall
    curve for each pair and calculate the average area under the curve
    """
    data = np.loadtxt(file, dtype=int, delimiter=",", skiprows=1)
    classes = [0, 1, 2, 3]
    # true = torch.t(torch.from_numpy(data[:,1]))
    # pred = F.one_hot(torch.from_numpy(data[:, 2]),num_classes = len(classes))
    # print(pred)
    # print("********")
    # average_precision = AveragePrecision(num_classes=len(classes))
    # accuracy = average_precision(pred, true)
    # preds=[]
    # target=[]
    # for i in range(len(true)):
    #     preds.append(dict())

    # metric = MeanAveragePrecision()
    # metric.update(pred,true)
    # m = metric.compute()
    # accuracy = m[0]
    # true = torch.from_numpy(data[:, 1])
    # pred = F.one_hot(torch.from_numpy(data[:, 2]),num_classes = len(classes))
    # pr_curve = PrecisionRecallCurve(num_classes= len(classes),pos_label=1)
    # precision, recall, thresholds = pr_curve(pred, true)
    # print(precision)
    # print(recall)
    # accuracy = auc(torch.tensor(precision),torch.tensor(recall))
    true = label_binarize(data[:, 1], classes=classes)
    pred = label_binarize(data[:, 2], classes=classes)
    precision = dict()
    recall = dict()
    accuracies = []
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(true[:, i], pred[:, i])
        accuracies.append(auc(recall[i],precision[i]))
        # accuracies.append(average_precision_score(true, pred))

    # # print(accuracies)
    accuracy = np.mean(accuracies)
    # precision, recall, thresholds = precision_recall_curve(true, pred)
    # accuracy = auc(precision, recall)
    return accuracy


def pr_auc(truefile, probfile):
    """
    PR-AUC is the area under the precision-recall curve.
    truefile = labels of data (Nx1)
    probfile = probabality sample belongs to each class (Nx3)
    """
    classes = [0, 1, 2]
    labels = np.load(truefile, allow_pickle=True)
    probs = np.loadtxt(probfile,dtype=float)
    #convert labels into Nx3 binary matrix
    labels = label_binarize(labels, classes=classes)
    precision = dict()
    recall = dict()
    threshold =dict()
    accuracies = []
    plt.axis()
    for i in range(len(classes)):
        precision[i], recall[i], threshold[i] = precision_recall_curve(labels[:, i], probs[:, i])
        accuracies.append(auc(recall[i],precision[i]))
        plt.plot(recall[i],precision[i],label=f"class {i}")
    plt.legend()
    figname = os.path.splitext(probfile)[0]
    plt.title(figname)
    plt.savefig(figname+'.png')
    plt.show()

    accuracy = np.mean(accuracies)
    # print(f"length of PR should be {len(classes)} and is {len(precision)}."+
    #     f"Num thresholds for each class is {len(threshold[0])}. "
    #     f"Num precision for each class is {len(precision[0])}")

    return accuracy


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
    if len(sys.argv) < 2:
        print("***Please provide model being assessed***")
        sys.exit()

    MODEL = sys.argv[1]
    if MODEL == "nnsplice":
        flanking = ["80nt","400nt"]
        for f in flanking:
            outputfile = "output/"+f+"_merged_nn_preds_0.csv"
            print(f"{outputfile} Accuracy is {simple_nn_accuracy(outputfile)}")
            print(f"and PR-AUC = {nn_pr_auc(outputfile)}")
            
    elif MODEL == "ourmodel":
        # print("***stay tuned***")
        truef = "datasets/processed/80nt_dev_labels.npy"
        classifiers = ["AdaBoost","CNN","SVM"]
        flanking = ["80nt"]
        for f in flanking:
            for c in classifiers:
                predf = "predictions/"+f+"_"+c+"_dev_predictions.csv"
                predf2 = "predictions/"+f+"_"+c+"_dev_softpredictions.csv"
                print(f"{predf} simple accuracy={simple_accuracy(truef,predf)}")
                print(f"{predf2} pr-auc={pr_auc(truef,predf2)}")
    else:
        print("*** no valid model given ***")
