import model_CNN


def train():
    # TODO: Train the model
    raise NotImplementedError()


def predict():
    # TODO: Predict
    raise NotImplementedError()


if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get("mode")

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise Exception("Mode not recognized")
