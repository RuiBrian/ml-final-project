from distutils.log import ERROR
from xml.etree.ElementPath import prepare_descendant
import numpy as np 
import sys
from sklearn.metrics import precision_recall_curve,auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
import torch.nn as nn

def simple_nn_accuracy(file):
    data = np.loadtxt(file, dtype=int,delimiter = ",",skiprows=1)
    true = data[:,1]
    pred = data[:,2]
    comparison = (true==pred)
    # incorrect = np.where(true!=pred)
    # print(len(incorrect[0]))
    # print(len(true))
    # accuracy = len(incorrect[0])/len(true)
    accuracy = ((np.sum(comparison))/comparison.size).astype(np.float64)
    return accuracy 

def simple_accuracy(truefile,predfile):
    labels = np.load(truefile,allow_pickle=True)
    preds = np.loadtxt(predfile)
    if len(labels) > len(preds):
        # print("labels len and pred len doesn't match")
        # return -1
        labels = labels[:len(preds)]
    # print(f"{len(labels)} and t{len(preds)}")
    comparison = (labels==preds)
    # incorrect = np.where(comparison=False)
    # print(incorrect)
    # accuracy = len(incorrect[0])/len(labels)
    accuracy = ((np.sum(comparison))/comparison.size).astype(np.float64)
    return accuracy 

def nn_pr_auc(file):
    """ 
    PR-AUC is the area under the precision-recall curve. 
    Binarize predictions and labels then calculate precision-recall 
    curve for each pair and calculate the average area under the curve
    """
    data = np.loadtxt(file, dtype=int,delimiter = ",",skiprows=1)
    classes=[0,1,2,3]
    true = label_binarize(data[:,1],classes=classes)
    pred = label_binarize(data[:,2],classes=classes)
    precision = dict()
    recall = dict()
    accuracies=[]
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(true[:, i],
                                                            pred[:, i])
        accuracies.append(auc(precision[i],recall[i]))
    # print(accuracies)
    accuracy = np.nanmean(accuracies)
    # precision, recall, thresholds = precision_recall_curve(true, pred)
    # accuracy = auc(precision, recall)
    return accuracy

def pr_auc(truefile,predfile):
    """ 
    PR-AUC is the area under the precision-recall curve. 
    """
    classes=[0,1,2]
    labels = np.load(truefile,allow_pickle=True)
    preds = np.loadtxt(predfile)
    if len(labels) > len(preds):
        # print("labels len and pred len doesn't match")
        labels = labels[:len(preds)]
    # print(f"{len(labels)} and t{len(preds)}")
    labels = label_binarize(labels,classes=classes)
    preds = label_binarize(preds,classes = classes)

    precision = dict()
    recall = dict()
    accuracies=[]
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            preds[:, i])
        accuracies.append(auc(precision[i],recall[i]))
    accuracy = np.nanmean(accuracies)
    return accuracy

def nn_top_k_accuracy(file):
    #TODO
    """
    Top-k accuracy is the fraction of correctly predicted splice sites 
    at the threshold where the number of predicted sites is equal to 
    the actual number of sites present.
    """
    raise NotImplementedError()

def top_k_accuracy(truefile,predfile):
    #TODO
    """
    Top-k accuracy is the fraction of correctly predicted splice sites 
    at the threshold where the number of predicted sites is equal to 
    the actual number of sites present.
    """
    raise NotImplementedError()


def accuracy(y : np.ndarray, y_hat : np.ndarray) -> np.float64: #printIdxs = False) -> np.float64:
    """Calculate the simple accuracy given two numpy vectors, each with int values
    corresponding to each class.

    Args:
        y (np.ndarray): actual value
        y_hat (np.ndarray): predicted value

    Returns:
        np.float64: accuracy
    """
    ### TODO Implement accuracy function
    # true if y and y_hat are equal
    comparison = (y==y_hat)
    # accuracy = number of true divided by total size of array
    accuracy = ((np.sum(comparison))/comparison.size).astype(np.float64)
    return accuracy 

def approx_train_acc_and_loss(model, train_data : np.ndarray, train_labels : np.ndarray) -> np.float64:
    """Given a model, training data and its associated labels, calculate the simple accuracy when the 
    model is applied to the training dataset.
    This function is meant to be run during training to evaluate model training accuracy during training.

    Args:
        model (pytorch model): model class object.
        train_data (np.ndarray): training data
        train_labels (np.ndarray): training labels

    Returns:
        np.float64: simple accuracy
    """
    idxs = np.random.choice(len(train_data), 4000, replace=False)
    x = torch.from_numpy(train_data[idxs].astype(np.float32))
    y = torch.from_numpy(train_labels[idxs].astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred.numpy()), loss.item()

def dev_acc_and_loss(model, dev_data : np.ndarray, dev_labels : np.ndarray, printIdxs=False) -> np.float64:
    """Given a model, a validation dataset and its associated labels, calcualte the simple accuracy when the
    model is applied to the validation dataset.
    This function is meant to be run during training to evaluate model validation accuracy.

    Args:
        model (pytorch model): model class obj
        dev_data (np.ndarray): validation data
        dev_labels (np.ndarray): validation labels

    Returns:
        np.float64: simple validation accuracy
    """
    x = torch.from_numpy(dev_data.astype(np.float32))
    y = torch.from_numpy(dev_labels.astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]

    return accuracy(dev_labels, y_pred.numpy()), loss.item()#dev_labels, y_pred.numpy(),printIdxs), loss.item()
 
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("***Please provide model being assessed***")
        sys.exit()
    
    MODEL = sys.argv[1]
    if MODEL =="nnsplice":
        outputfile = "output/merged_nn_preds0.csv"
        if len(sys.argv) > 2:
            outputfile = sys.argv[2]
        print(f"{outputfile} Accuracy is {simple_nn_accuracy(outputfile)*100:.3f}% and PR-AUC = {nn_pr_auc(outputfile)*100:.3f}%")    
    elif MODEL =="ourmodel":
        # print("***stay tuned***")
        truef = "datasets/processed/dev_labels.npy"
        predf = "predictions/CNN_predictions.csv"
        if len(sys.argv) > 2:
            truef = sys.argv[2]
            if len(sys.argv) > 3:
                predf = sys.argv[3]
        print(f"simple accuracy={simple_accuracy(truef,predf)} and pr-auc={pr_auc(truef,predf)}")

    else:
        print("*** no valid model given ***")