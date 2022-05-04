from model_CNN import CNN
import torch
import torch.nn.functional as F
import numpy as np
import sys
from torchsummary import summary


def train(device, num_NT):
    
    TRAIN_SEQUENCES = np.load("datasets/processed/{}nt_train_encoded.npy".format(num_NT))
    TRAIN_LABELS = np.load("datasets/processed/{}nt_train_labels.npy".format(num_NT))

    # Number of gene sequences in the training corpus
    N_SEQUENCES = TRAIN_LABELS.shape[0]

    # Dimensions of a one-hot encoded sequence
    HEIGHT = int(num_NT) + 2
    WIDTH = 4

    # Number of output classes
    N_CLASSES = 3

    # Parameters
    LEARNING_RATE = 0.001
    EPOCHS = 28000

    # Initialize model and optimizer
    model = CNN(input_height=HEIGHT, input_width=WIDTH, n_classes=N_CLASSES)
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

    torch.save(model, "models/{}nt_CNN.pt".format(num_NT))


def predict(device, num_NT):
    
    DEV_SEQUENCES = np.load("datasets/processed/{}nt_dev_encoded.npy".format(num_NT))
    HEIGHT = int(num_NT) + 2
    NUM_SEQUENCES = int(DEV_SEQUENCES.shape[0] / HEIGHT)

    model = torch.load("models/CNN.pt")
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

    predictions = np.array(predictions)
    soft_predictions = np.array(soft_predictions)
    
    # print(soft_predictions)
    # print(f"pred {len(predictions)} sp {len(soft_predictions)} ")
    # np.savetxt("predictions/{}nt_CNN_dev_predictions.csv".format(num_NT), predictions, fmt="%d")
    # np.savetxt("predictions/{}nt_CNN_dev_softpredictions.csv".format(num_NT), soft_predictions, fmt="%f")


if __name__ == "__main__":
    MODE = sys.argv[1]
    num_NT = sys.argv[2]

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if MODE == "train":
        train(device, num_NT)
    elif MODE == "predict":
        predict(device, num_NT)
    else:
        raise Exception("Mode not recognized")
