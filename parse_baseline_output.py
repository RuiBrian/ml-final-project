from html.parser import HTMLParser
import sys
import os
import numpy as np
import csv
from bs4 import BeautifulSoup

# TODO make output just predicted coumn
def parse_nn_html(file):
    """
    0 = neither
    1 = donor
    2 = acceptor
    3 = both
    """
    oldpath = os.getcwd()
    os.chdir("output")

    # find true label based on input file
    if "donor" in file:
        truelabel = 1
    elif "acceptor" in file:
        truelabel = 2
    else:
        truelabel = 0

    # start array to store csv data
    data = np.empty((0, 3), int)

    # open html file and get raw predictions
    HTMLFile = open(file, "r")
    index = HTMLFile.read()
    Parse = BeautifulSoup(index, "html.parser")
    ## ***** this is literally so shitty wtf improve this *****
    preelements = Parse.find_all("pre")
    empty_label_len_donor = len("Start   End    Score     Exon   Intron")
    empty_label_len_acceptor = len("Start   End    Score     Intron               Exon")
    tag_idx = 0
    line_idx = 0
    for h in Parse.find_all("h3"):
        ht = h.get_text()
        # print(ht)
        if "Donor site predictions for line" in ht:
            line_idx = int(ht[-3])
            # print(line_idx)
            break
    newfileidx1 = line_idx

    # categorize predictions
    predlabel = 0
    for p in preelements:
        if p.text:
            # print(p.text)
            if tag_idx % 2 == 0:  # donor predictions comes first
                if (
                    len(p.text) > empty_label_len_donor
                ):  # if a site prediction is present, mark sequence as donor
                    predlabel = 1
                else:  # otherwise say sequence is neither
                    predlabel = 0
            else:  # then acceptor predictions
                if len(p.text) > empty_label_len_acceptor:
                    if not predlabel:
                        predlabel = 2
                    else:  # if model predicts both donor and acceptor, model classified sequence as both
                        predlabel = truelabel  # 3
                data = np.append(data, [[line_idx, truelabel, predlabel]], axis=0)
                line_idx += 1
            tag_idx += 1
    # print(f'data {np.shape(data)}')

    # write outputs to a new csv file
    newfileidx2 = line_idx
    newfileprefix = os.path.splitext(file)[0]
    newfile = newfileprefix + "_" + str(newfileidx1) + "_" + str(newfileidx2) + ".csv"
    headers = ["line num", "true label", "predicted label"]

    with open(newfile, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(data)
    os.chdir(oldpath)
    print(f"created {newfile} with {newfileidx2-newfileidx1} preds")
    return newfile


def merge_nn_output(files, descriptor=""):
    headers = ["line num", "true label", "predicted label"]
    data = np.empty((0, 3), int)
    for f in files:
        # print(f"{data} and {np.shape(data)}")
        # print(np.loadtxt(f, dtype=int,delimiter = ",",skiprows=1))
        data = np.append(
            data, np.loadtxt(f, dtype=int, delimiter=",", skiprows=1), axis=0
        )
    # print(np.shape(data))
    suffix = 0
    descriptor = descriptor + "_"
    newfile = f"output/{descriptor}merged_nn_preds_{suffix}.csv"
    while os.path.exists(newfile):
        suffix += 1
        newfile = f"output/merged_nn_preds{suffix}.csv"
    with open(newfile, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("***Please specify baseline (nnsplice or spliceai)***")
    BASELINE = sys.argv[1]
    if BASELINE == "nnsplice":
        if len(sys.argv) < 3:
            print("*** input flanking sequence length ***")
            sys.exit()
        else:
            des = sys.argv[2]
            if len(sys.argv) == 4:
                hfile = sys.argv[3]
                if ".csv" not in hfile:
                    parse_nn_html(hfile)
                else:
                    print("invalid raw output given (must be html)")
            else:
                hfiles = os.listdir("output")
                parsed = []
                for h in hfiles:
                    if "nnsplice_" in h and des in h and ".csv" not in h:
                        parsed.append("output/" + parse_nn_html(h))
                # print(parsed)
                # merge into one file
                merge_nn_output(parsed, descriptor=des)

    elif BASELINE == "spliceai":
        print("this doesn't do anything")
