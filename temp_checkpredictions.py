import numpy as np
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn

dev_encoded = np.load('datasets/processed/dev_encoded.npy')
print("Total dev test: ", int(dev_encoded.shape[0]/82))

dev_labels = np.load('datasets/processed/dev_labels.npy')
print("Total true label=0: ", np.sum(dev_labels==0))
print("Total true label=1: ", np.sum(dev_labels==1))
print("Total true label=2: ", np.sum(dev_labels==2))

dev_predict = np.loadtxt("predictions/CNN_dev_predictions.csv", dtype=int)
print("Total predict label=0: ", np.sum(dev_predict==0))
print("Total predict label=1: ", np.sum(dev_predict==1))
print("Total predict label=2: ", np.sum(dev_predict==2))

print("Accuracy: ", np.sum(np.transpose(dev_labels)[0]==dev_predict)/int(dev_encoded.shape[0]/82))

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




