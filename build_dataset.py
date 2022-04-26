import pandas as pd


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

    # TODO: Split into train/dev/test (70%/20%/10%)


if __name__ == "__main__":
    build_dataset()
