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

Run model: Copy and paste 1000 sequences to the website and submit. Save the HTML file.

Scrape NNSplice output:
```shell
$ python parse_baseline_output.py nnsplice [flanking sequence length]
```

Compute accuracy:
```shell
$ python accuracies.py nnsplice path_to_output.csv[default=output/merged_nn_preds0.csv]
```

## CNN model

Train the model:

```shell
$ python main_CNN.py train num_NT[default=80]
```

Predict with the model:

```shell
$ python main_CNN.py predict num_NT[default=80]
```

Build dataset:
```shell
$ python build_dataset.py
```

## SVM

Fit and predict:

```shell
$ python main_SVM.py num_NT[default=80]
```

## AdaBoost


Fit and predict:

```shell
$ python main_AdaBoost.py num_NT[default=80]
```

## Compute Accuracy

```shell
$ python accuracies.py [nnsplice or ourmodel] [80 or 400]
```
