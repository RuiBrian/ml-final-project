import pandas as pd
import numpy as np
import sys
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(file, flanking_seq):
    # Prevent truncation of long strings
    pd.set_option("display.max_colwidth", None)

    # Load file into dataframe
    df = pd.read_csv(
        f"datasets/processed/{flanking_seq}nt_{file}.csv", header=None, index_col=False
    )

    # Split labels and sequences
    labels_df = df.iloc[:, [0]]
    sequences_df = df.iloc[:, [1]]

    # Create a column for each letter in sequence
    sequences_df = sequences_df.apply(
        lambda seq: pd.Series(",".join(list(seq.to_string(index=False))).split(",")),
        axis=1,
    )

    sequences_df.to_csv(
        f"datasets/processed/{flanking_seq}nt_{file}_separated.csv",
        header=None,
        index=False,
    )

    # Perform one-hot encoding with pd.get_dummies()
    sequences_df = sequences_df.transpose()
    sequences_df = pd.Series(sequences_df.values.ravel("F"))
    sequences_df = pd.get_dummies(sequences_df)

    # Convert to numpy arrays and save as .npy
    np.save(
        f"datasets/processed/{flanking_seq}nt_{file}_labels.npy", labels_df.to_numpy()
    )
    np.save(
        f"datasets/processed/{flanking_seq}nt_{file}_encoded.npy",
        sequences_df.to_numpy(),
    )


def build_dataset(flanking_seq):
    # Load into dataframes
    neither_df = pd.read_csv(
        f"datasets/raw/{flanking_seq}nt_neither.txt", header=None, index_col=False
    )
    # exons_df = pd.read_csv(
    #     f"datasets/raw/{flanking_seq}nt_exons.txt", header=None, index_col=False
    # )
    donors_df = pd.read_csv(
        f"datasets/raw/{flanking_seq}nt_donors.txt", header=None, index_col=False
    )
    acceptors_df = pd.read_csv(
        f"datasets/raw/{flanking_seq}nt_acceptors.txt", header=None, index_col=False
    )

    # Sample smaller portions of dataframe (50% for 80nt and 10% for 400nt)
    frac_totaldata = 0.5 if flanking_seq == 80 else 0.1
    neither_df = neither_df.sample(frac=frac_totaldata, replace=False, random_state=0)
    # exons_df = exons_df.sample(frac=frac_totaldata / 2, replace=False, random_state=0)
    donors_df = donors_df.sample(frac=frac_totaldata, replace=False, random_state=0)
    acceptors_df = acceptors_df.sample(
        frac=frac_totaldata, replace=False, random_state=0
    )

    # Convert to all uppercase
    neither_df = neither_df.applymap(lambda s: s.upper())
    # exons_df = exons_df.applymap(lambda s: s.upper())
    donors_df = donors_df.applymap(lambda s: s.upper())
    acceptors_df = acceptors_df.applymap(lambda s: s.upper())

    # Concat neither_filtered.fa and exons.fa and label as neither (0)
    # neither_df = pd.concat([neither_df, exons_df])
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
    train.to_csv(
        f"datasets/processed/{flanking_seq}nt_train.csv", header=None, index=False
    )
    dev.to_csv(f"datasets/processed/{flanking_seq}nt_dev.csv", header=None, index=False)
    test.to_csv(
        f"datasets/processed/{flanking_seq}nt_test.csv", header=None, index=False
    )


def nnsplice_dataset(
    filename, inputdir="datasets/raw/", outputdir="datasets/nnsplice/"
):
    if not filename.endswith(".fa"):
        print(f"incorrect file {filename}")
        return -1
    # add fasta title to separate each sequence
    newfilename = outputdir + "nnsplice_" + filename
    startChar = ">"
    i = 0
    with open(inputdir + filename, "r") as sequences:
        with open(newfilename, "w") as updateFile:
            for seq in sequences:
                if seq[0] != startChar:
                    updateFile.write(
                        startChar + f"line{i} len={len(seq)-1}\n" + seq.rstrip() + "\n"
                    )
                    i += 1
    print(newfilename)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("***Please specify dataset mode (nnsplice or ourmodel)***")
        sys.exit()

    MODE = sys.argv[1]

    if MODE == "nnsplice":
        hfiles = os.listdir("datasets/raw")
        for h in hfiles:
            nnsplice_dataset(h)

        # nnsplice_dataset("acceptors.fa")
        # nnsplice_dataset("donors.fa")
        # nnsplice_dataset("exons.fa")
        # nnsplice_dataset("neither.fa")

    elif MODE == "ourmodel":
        if len(sys.argv) < 3:
            print("***Please specify number of nucleotides (80 or 400)***")
            sys.exit()

        flanking_seq = int(sys.argv[2])

        if flanking_seq != 80 and flanking_seq != 400:
            print("***Invalid number of nucleotides***")
            sys.exit()

        build_dataset(flanking_seq)

        # Perform one-hot encoding on each split
        one_hot_encode("train", flanking_seq)
        one_hot_encode("dev", flanking_seq)
        one_hot_encode("test", flanking_seq)
    else:
        print("***Mode not recognized***")
