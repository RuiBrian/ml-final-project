import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(file):
    # Prevent truncation of long strings
    pd.set_option("display.max_colwidth", None)

    # Load file into dataframe
    df = pd.read_csv(f"datasets/processed/{file}.csv", header=None, index_col=False)

    # Split labels and sequences
    labels_df = df.iloc[:, [0]]
    sequences_df = df.iloc[:, [1]]

    # Create a column for each letter in sequence
    sequences_df = sequences_df.apply(
        lambda seq: pd.Series(",".join(list(seq.to_string(index=False))).split(",")),
        axis=1,
    )

    # TODO: One-hot encoding

    # Output labels and encoded sequences to separate CSV files
    labels_df.to_csv(f"datasets/processed/{file}_labels.csv", header=None, index=False)


def build_dataset():
    # Load into dataframes
    neither_df = pd.read_csv("datasets/raw/neither.fa", header=None, index_col=False)
    exons_df = pd.read_csv("datasets/raw/exons.fa", header=None, index_col=False)
    donors_df = pd.read_csv("datasets/raw/donors.fa", header=None, index_col=False)
    acceptors_df = pd.read_csv(
        "datasets/raw/acceptors.fa", header=None, index_col=False
    )

    # Convert to all uppercase
    neither_df = neither_df.applymap(lambda s: s.upper())
    exons_df = exons_df.applymap(lambda s: s.upper())
    donors_df = donors_df.applymap(lambda s: s.upper())
    acceptors_df = acceptors_df.applymap(lambda s: s.upper())

    # Concat neither_filtered.fa and exons.fa and label as neither (0)
    neither_df = pd.concat([neither_df, exons_df])
    neither_df.insert(0, "label", 0)

    # Label donors (1)
    donors_df.insert(0, "label", 1)

    # Label acceptors (2)
    acceptors_df.insert(0, "label", 2)

    # Concat all
    all_df = pd.concat([neither_df, donors_df, acceptors_df])

    # Split into train/dev/test (60%/20%/20%)
    train, dev, test = np.split(
        all_df.sample(frac=1, random_state=0),
        [int(0.6 * len(all_df)), int(0.8 * len(all_df))],
    )

    # Print number of examples for each split
    print(f"Number of training examples: {train.shape[0]}")
    print(f"Number of dev examples: {dev.shape[0]}")
    print(f"Number of test examples: {test.shape[0]}")

    # Write to CSV files
    train.to_csv("datasets/processed/train.csv", header=None, index=False)
    dev.to_csv("datasets/processed/dev.csv", header=None, index=False)
    test.to_csv("datasets/processed/test.csv", header=None, index=False)


if __name__ == "__main__":
    build_dataset()

    # Perform one-hot encoding on each split
    one_hot_encode("train")
    one_hot_encode("dev")
    one_hot_encode("test")
