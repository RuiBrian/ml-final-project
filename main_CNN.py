import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import csv

from model_CNN import CNN
from sklearn import metrics

def train(device, flanking_seq):
    
    # Load train dataset and labels
    TRAIN_SEQUENCES = np.load(
        f"datasets/processed/{flanking_seq}nt_train_encoded.npy"
        )
    TRAIN_LABELS = np.load(
        f"datasets/processed/{flanking_seq}nt_train_labels.npy"
        )

    # Number of gene sequences in the training corpus
    N_SEQUENCES = TRAIN_LABELS.shape[0]

    # Dimensions of a one-hot encoded sequence
    HEIGHT = int(flanking_seq) + 2
    WIDTH = 4

    # Number of output classes
    N_CLASSES = 3

    # Parameters
    LEARNING_RATE = 0.0001
    # TODO: Ephocs not used
    EPOCHS = 500000
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")

    # Initialize model and optimizer
    model = CNN(input_height=HEIGHT, input_width=WIDTH, n_classes=N_CLASSES)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    s = torch.nn.Softmax(dim=1)
    cur_label_index = 0
    for step in range(0, TRAIN_SEQUENCES.shape[0], HEIGHT):
        x = torch.from_numpy(
            TRAIN_SEQUENCES[step : step + HEIGHT].astype(np.float32)
            ).to(device)
        y = torch.from_numpy(TRAIN_LABELS[cur_label_index].astype(int)
            ).to(device)

        # Forward pass: get logits for x
        logits = model(x)
        
        # Compute soft prediction
        soft_pred = s(logits).detach().cpu().numpy().flatten()

        # Compute loss
        loss = F.cross_entropy(logits, y)

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy and loss
        y_hat = torch.max(logits, 1)[1]
        train_acc = soft_pred[y]
        train_loss = loss.item()
        
        step_metrics = {
            'step': cur_label_index,
            'train_loss': train_loss,
            'train_acc': train_acc,
        }
        
        logger.writerow(step_metrics)
        # TODO: delete loss or not?
        del loss
        cur_label_index += 1
        
    torch.save(model, f"models/{flanking_seq}nt_CNN.pt")
    LOGFILETRAIN.close()


def predict(device, flanking_seq, dev_or_test):
    
    # Predict on dev or test set
    dev_or_test = dev_or_test
    
    # Load dataset and labels
    DEV_SEQUENCES = np.load(
        f"datasets/processed/{flanking_seq}nt_{dev_or_test}_encoded.npy")
    DEV_LABELS = np.load(
        f"datasets/processed/{flanking_seq}nt_{dev_or_test}_labels.npy")
    
    # Dimensions of a one-hot encoded sequence
    HEIGHT = int(flanking_seq) + 2
    
    # Load the model
    model = torch.load(f"models/{flanking_seq}nt_CNN.pt")
    model.to(device)
    
    # Initialize predictions and soft_predictions
    predictions = []
    soft_predictions = []

    s = torch.nn.Softmax(dim=1)
    cur_label_index = 0
    for i in range(0, DEV_SEQUENCES.shape[0], HEIGHT):
        x = torch.from_numpy(DEV_SEQUENCES[i : i + HEIGHT].astype(np.float32)
            ).to(device)
        y = torch.from_numpy(DEV_LABELS[cur_label_index].astype(int)
            ).to(device)
        
        # Forward pass: get logits for x
        logits = model(x)
        
        # Compute soft prediction for PR curve
        soft_pred = s(logits).detach().cpu().numpy().flatten()
        soft_predictions.append(soft_pred)
        
        # Compute accuracy and loss
        loss = F.cross_entropy(logits, y)
        pred = torch.max(logits, 1)[1]
        predictions.append(pred.item())
        # TODO: acc = soft_pred[y]?
        dev_acc = soft_pred[y]
        dev_loss = loss.item()
        
        if dev_or_test == "dev":
            step_metrics = {
                'step': cur_label_index,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc,
            }
        elif dev_or_test == "test":
            step_metrics = {
                'step': cur_label_index,
                'test_loss': dev_loss,
                'test_acc': dev_acc,
            }
        
        logger.writerow(step_metrics)
    
        # TODO: delete loss or not?
        del loss
        cur_label_index += 1
              
    predictions = np.array(predictions)
    soft_predictions = np.array(soft_predictions)
    
    if dev_or_test == "dev":
        LOGFILEDEV.close()
    elif dev_or_test == "test":
        LOGFILETEST.close()
    
    # Save predictions and soft_predictions
    np.savetxt(
        f"predictions/{flanking_seq}nt_CNN_{dev_or_test}_predictions.csv", 
        predictions, fmt="%d"
        )
    np.savetxt(
        f"predictions/{flanking_seq}nt_CNN_{dev_or_test}_softpredictions.csv",
        soft_predictions,
        fmt="%f",
        )
    print(
        f"Accuracy: {metrics.accuracy_score(DEV_LABELS.ravel(), predictions)}"
        )


if __name__ == "__main__":
    MODE = sys.argv[1]

    if len(sys.argv) == 3:
        flanking_seq = int(sys.argv[2])
    elif len(sys.argv) == 2:
        flanking_seq = 80

    if flanking_seq != 80 and flanking_seq != 400:
        raise Exception("Flanking sequence must be 80 or 400")

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if MODE == "train":
        # Write logging model performance for training set
        LOGFILETRAIN = open(os.path.join(
            f"logs/{flanking_seq}nt_train_CNN.log"), 'w'
            )
        log_fieldnames = ['step', 'train_loss', 'train_acc']
        logger = csv.DictWriter(LOGFILETRAIN, log_fieldnames)
        logger.writeheader()
        
        train(device, flanking_seq)
        
        print("Model trained \n Predicting dev set...")
        
        # Write logging model performance for dev set
        LOGFILEDEV = open(os.path.join(
            f"logs/{flanking_seq}nt_dev_CNN.log"), 'w'
            )
        log_fieldnames = ['step', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILEDEV, log_fieldnames)
        logger.writeheader()
        
        predict(device, flanking_seq, "dev")
        
    elif MODE == "predict":
        # Write logging model performance
        LOGFILETEST = open(os.path.join(
            f"logs/{flanking_seq}nt_test_CNN.log"), 'w'
            )
        log_fieldnames = ['step', 'test_loss', 'test_acc']
        logger = csv.DictWriter(LOGFILETEST, log_fieldnames)
        logger.writeheader()
        
        predict(device, flanking_seq, "test")
        
    else:
        raise Exception("Mode not recognized")
