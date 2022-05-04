import numpy as np
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score

num_NT = 80

dev_encoded = np.load('datasets/processed/{}nt_dev_encoded.npy'.format(num_NT))
print("Total dev test: ", int(dev_encoded.shape[0]/82))

dev_labels = np.load('datasets/processed/{}nt_dev_labels.npy'.format(num_NT))
print("Total true label=0: ", np.sum(dev_labels==0))
print("Total true label=1: ", np.sum(dev_labels==1))
print("Total true label=2: ", np.sum(dev_labels==2))

dev_predict = np.loadtxt("predictions/{}nt_CNN_dev_predictions.csv".format(num_NT), dtype=int)
print("Total predict label=0: ", np.sum(dev_predict==0))
print("Total predict label=1: ", np.sum(dev_predict==1))
print("Total predict label=2: ", np.sum(dev_predict==2))
print("Accuracy: ", np.sum(np.transpose(dev_labels)[0]==dev_predict)/int(dev_encoded.shape[0]/82))
# print(f"Accuracy: {accuracy_score(dev_labels, dev_predict)}")

# dev_softpredict = np.loadtxt("predictions/CNN_dev_softpredictions.csv", dtype=np.float32)
# max_softpredict = dev_softpredict.max(axis = 1)
# print("Total softpredict label=0: ", np.sum(dev_softpredict[:,0] == max_softpredict))
# print("Total softpredict label=1: ", np.sum(dev_softpredict[:,1] == max_softpredict))
# print("Total softpredict label=2: ", np.sum(dev_softpredict[:,2] == max_softpredict))


# TRAIN_SEQUENCES = np.load("datasets/processed/train_encoded.npy")
# print(TRAIN_SEQUENCES.shape)
# step = 0
# x1 = torch.from_numpy(TRAIN_SEQUENCES[step : step + 82].astype(np.float32))
# print(x1.shape)
# # x2 = torch.from_numpy(TRAIN_SEQUENCES[step+82 : step+82 + 82].astype(np.float32))
# # for i in range(len(x1)-1):
# #     print(x1[i+1]==x2[i])
    
# TRAIN_LABELS = np.load("datasets/processed/train_labels.npy")
# # print(TRAIN_LABELS.shape)
# # print(TRAIN_SEQUENCES.shape[0]/TRAIN_LABELS.shape[0])