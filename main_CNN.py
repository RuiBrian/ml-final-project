import torch
import torch.nn.functional as F
import numpy as np
import sys

from model_CNN import CNN
from sklearn import metrics


def train(device, num_NT):
    TRAIN_SEQUENCES = np.load(f"datasets/processed/{num_NT}nt_train_encoded.npy")
    TRAIN_LABELS = np.load(f"datasets/processed/{num_NT}nt_train_labels.npy")

    # Number of gene sequences in the training corpus
    N_SEQUENCES = TRAIN_LABELS.shape[0]

    # Dimensions of a one-hot encoded sequence
    HEIGHT = int(num_NT) + 2
    WIDTH = 4

    # Number of output classes
    N_CLASSES = 3

    # Parameters
    LEARNING_RATE = 0.0001
    EPOCHS = 500000
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")

    # Initialize model and optimizer
    model = CNN(input_height=HEIGHT, input_width=WIDTH, n_classes=N_CLASSES)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    cur_label_index = 0
    for step in range(0, EPOCHS, HEIGHT):
        x = torch.from_numpy(
            TRAIN_SEQUENCES[step : step + HEIGHT].astype(np.float32)
        ).to(device)
        y = torch.from_numpy(TRAIN_LABELS[cur_label_index].astype(int)).to(device)
        cur_label_index += 1

        # Forward pass: get logits for x
        logits = model(x)

        # Compute loss
        loss = F.cross_entropy(logits, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, f"models/{num_NT}nt_CNN.pt")


def predict(device, num_NT):
    DEV_SEQUENCES = np.load(f"datasets/processed/{num_NT}nt_dev_encoded.npy")
    HEIGHT = int(num_NT) + 2
    NUM_SEQUENCES = int(DEV_SEQUENCES.shape[0] / HEIGHT)

    model = torch.load(f"models/{num_NT}nt_CNN.pt")
    model.to(device)
    predictions = []
    soft_predictions = []

    s = torch.nn.Softmax(dim=1)
    for i in range(0, DEV_SEQUENCES.shape[0], HEIGHT):
        x = torch.from_numpy(DEV_SEQUENCES[i : i + HEIGHT].astype(np.float32)).to(
            device
        )
        logits = model(x)

        pred = torch.max(logits, 1)[1]
        soft_pred = s(logits).detach().numpy().flatten()
        predictions.append(pred.item())
        soft_predictions.append(soft_pred)

    true_labels = np.load(f"datasets/processed/{num_NT}nt_dev_labels.npy").ravel()
    predictions = np.array(predictions)
    soft_predictions = np.array(soft_predictions)
    np.savetxt(f"predictions/{num_NT}nt_CNN_dev_predictions.csv", predictions, fmt="%d")
    np.savetxt(
        f"predictions/{num_NT}nt_CNN_dev_softpredictions.csv",
        soft_predictions,
        fmt="%f",
    )
    print(f"Accuracy: {metrics.accuracy_score(true_labels, predictions)}")


if __name__ == "__main__":
    MODE = sys.argv[1]

    if len(sys.argv) == 3:
        num_NT = int(sys.argv[2])
    elif len(sys.argv) == 2:
        num_NT = 80
        
    if num_NT != 80 and num_NT != 400:
        raise Exception("Flanking sequence must be 80 or 400")

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if MODE == "train":
        train(device, num_NT)
    elif MODE == "predict":
        predict(device, num_NT)
    else:
        raise Exception("Mode not recognized")
