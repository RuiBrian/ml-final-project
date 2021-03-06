# ml-final-project

## Team Members

Brian Rui (brui1@jhu.edu)

Carolyna Yamamoto Alves Pinto (cyamamo2@jhu.edu)

Jakob Heinz (jheinz3@jhu.edu)

Riddhi Gopal (rgopal3@jhu.edu)

## Setup

Create a virtual environment:

```shell
$ python3 -m venv ml-final-project
```

Install dependencies:

```shell
$ pip install -r requirements.txt
```

Build dataset:

```shell
$ python build_dataset.py ourmodel
```

## NNSplice Baseline

Prepare dataset for NNsplice:

```shell
$ python build_dataset.py nnsplice
```

Run model: Copy and paste 1000 80nt sequences (or 230 400nt sequences) to the website and submit. Save the HTML file tp the output folder.

Scrape NNSplice output:
```shell
$ python parse_baseline_output.py nnsplice <flanking sequence length>
```

Compute accuracy:
```shell
$ python accuracies.py nnsplice <path_to_output.csv[default=output/merged_nn_preds0.csv]> <flanking_seq [80 or 400]>
```

## CNN model

Train the model:

```shell
$ python main_CNN.py train <flanking_seq [80 or 400]>
```

Predict with the model:

```shell
$ python main_CNN.py predict <flanking_seq [80 or 400]>
```

## SVM

Fit and predict:

```shell
$ python main_SVM.py <flanking_seq [80 or 400]>
```

## AdaBoost


Fit and predict:

```shell
$ python main_AdaBoost.py <flanking_seq [80 or 400]> <dataset [default=dev]>
```

## Compute Accuracy

```shell
$ python accuracies.py <nnsplice or ourmodel> <flanking_seq [80 or 400]>
```
