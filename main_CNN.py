from model_CNN import CNN
import torch
import torch.nn.functional as F
import numpy as np
import sys
from torchsummary import summary


def train():
    TRAIN_SEQUENCES = np.load("datasets/processed/train_encoded.npy")
    # TRAIN_SEQUENCES = (TRAIN_SEQUENCES - TRAIN_SEQUENCES.mean()) / (
    #     TRAIN_SEQUENCES.std()
    # )
    TRAIN_LABELS = np.load("datasets/processed/train_labels.npy")

    # Number of gene sequences in the training corpus
    N_SEQUENCES = TRAIN_LABELS.shape[0]

    # Dimensions of a one-hot encoded sequence
    HEIGHT = 82
    WIDTH = 4

    # Number of output classes
    N_CLASSES = 3

    # Parameters
    LEARNING_RATE = 0.0001
    EPOCHS = 10000

    # Initialize model and optimizer
    model = CNN(input_height=HEIGHT, input_width=WIDTH, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()
    
    s = 0
    for step in range(0, EPOCHS, HEIGHT):
        x = torch.from_numpy(TRAIN_SEQUENCES[step : step + HEIGHT].astype(np.float32))
        y = torch.from_numpy(TRAIN_LABELS[s].astype(int))
        s += 1
            
        # Forward pass: get logits for x
        logits = model(x)
        logits_label = torch.softmax(logits, 1)

        # Compute loss
        loss = F.cross_entropy(logits, y)
        
        if step < 100*82:
            # print(torch.softmax(logits, 1))
            print(torch.max(logits_label, 1)[1])

        # Zero gradients, perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()

    torch.save(model, "models/CNN.pt")


def predict():
    TEST_SEQUENCES = np.load("datasets/processed/test_encoded.npy")
    # TEST_SEQUENCES = (TEST_SEQUENCES - TEST_SEQUENCES.mean()) / (TEST_SEQUENCES.std())
    HEIGHT = 82
    NUM_SEQUENCES = int(TEST_SEQUENCES.shape[0] / HEIGHT)

    model = torch.load("models/CNN.pt")
    predictions = []

    for i in range(0, TEST_SEQUENCES.shape[0], HEIGHT):
        x = torch.from_numpy(TEST_SEQUENCES[i : i + HEIGHT].astype(np.float32))
        logits = model(x)
        
        pred_normalized = torch.softmax(logits, 1)
        pred = torch.max(pred_normalized, 1)[1]
        predictions.append(pred.item())
        
        # if i < 100*82:
        #     print(pred_normalized)

    predictions = np.array(predictions)
    np.savetxt("predictions/CNN_predictions.csv", predictions, fmt="%d")


if __name__ == "__main__":
    MODE = sys.argv[1]

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise Exception("Mode not recognized")
