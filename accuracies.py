from xml.etree.ElementPath import prepare_descendant
import numpy as np 
import sys
from sklearn.metrics import precision_recall_curve,auc
from sklearn.preprocessing import label_binarize

def simple_nn_accuracy(file):
    data = np.loadtxt(file, dtype=int,delimiter = ",",skiprows=1)
    true = data[:,1]
    pred = data[:,2]
    incorrect = np.where(true!=pred)
    # print(len(incorrect[0]))
    # print(len(true))
    accuracy = len(incorrect[0])/len(true)
    return accuracy 

def simple_accuracy(truefile,predfile):
    labels = np.loadtxt(truefile, dtype=int)
    preds = np.loadtxt(predfile,dtype=int)
    incorrect = np.where(labels!=preds)
    accuracy = len(incorrect[0])/len(labels)
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
    labels = label_binarize(np.loadtxt(truefile, dtype=int))
    preds = label_binarize(np.loadtxt(predfile,dtype=int))
    classes=[0,1,2,3]
    precision = dict()
    recall = dict()
    accuracies=[]
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            preds[:, i])
        accuracies.append(auc(precision[i],recall[i]))
    # accuracy = auc(precision, recall)
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

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("***Please provide model being assessed***")
        sys.exit()
    
    MODEL = sys.argv[1]
    if MODEL =="nnsplice":
        if len(sys.argv) < 3:
            print("***Please provide output file for nnsplice***")
        outputfile = sys.argv[2]
        print(f"{outputfile} Accuracy is {simple_nn_accuracy(outputfile)*100:.3f}% and PR-AUC = {nn_pr_auc(outputfile)*100:.3f}%")    
    elif MODEL =="ourmodel":
        print("***stay tuned***")