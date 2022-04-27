import numpy as np 
import sys

def simple_accuracy(file):
    data = np.loadtxt(file, dtype=int,delimiter = ",",skiprows=1)
    true = data[:,1]
    pred = data[:,2]
    incorrect = np.where(true!=pred)
    # print(len(incorrect[0]))
    # print(len(true))
    accuracy = len(incorrect[0])/len(true)
    return accuracy 

def pr_auc(file):
    #TODO 
    raise NotImplementedError()

def top_k_accuracy(file):
    #TODO
    raise NotImplementedError()

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("***Please provide path to output file to calculate accuracy for***")
        sys.exit()
    
    outputfile = sys.argv[1]
    print(f"Accuracy for {outputfile} is {simple_accuracy(outputfile)*100:.3f}%")
