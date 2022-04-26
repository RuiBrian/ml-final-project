import pandas as pd
import numpy as np


def build_dataset():
    # Load into dataframes
    neither_df = pd.read_csv("datasets/raw/neither.fa",
                             header=None, index_col=False)
    exons_df = pd.read_csv("datasets/raw/exons.fa",
                           header=None, index_col=False)
    donors_df = pd.read_csv("datasets/raw/donors.fa",
                            header=None, index_col=False)
    acceptors_df = pd.read_csv(
        "datasets/raw/acceptors.fa", header=None, index_col=False)

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

    # TODO: One-hot encoding

    # Split into train/dev/test (60%/20%/20%)
    train, dev, test = np.split(all_df.sample(frac=1, random_state=0),
                                [int(0.6 * len(all_df)), int(0.8 * len(all_df))])

    # Print number of examples for each split
    print(f"Number of training examples: {train.shape[0]}")
    print(f"Number of dev examples: {dev.shape[0]}")
    print(f"Number of test examples: {test.shape[0]}")

    # Write to CSV files
    train.to_csv("datasets/train/train.csv", header=None, index=False)
    dev.to_csv("datasets/dev/dev.csv", header=None, index=False)
    test.to_csv("datasets/test/test.csv", header=None, index=False)


if __name__ == "__main__":
    build_dataset()
