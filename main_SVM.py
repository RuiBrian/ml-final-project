import numpy as np
from sklearn.svm import SVC


def fit():
    raise NotImplementedError()


def predict():
    raise NotImplementedError()


if __name__ == "__main__":
    MODE = sys.argv[1]

    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise Exception("Mode not recognized")
