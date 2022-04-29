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
$ python build_dataset.py
```
## NNSplice Baseline
Prepare dataset for NNsplice:
```shell
$ python python build_dataset.py nnsplice
```
Run model:
Copy paste 1000 sequences to the website and submit
save html file 

Scrape NNSplice output:
```shell
$ python parse_baseline_output.py nnsplice
```
Compute accuracy:
```shell
$ python accuracies.py nnsplice
```

## Running the model

Train the model:

```shell
$ python main_CNN.py train
```

Predict with the model:

```shell
$ python main_CNN.py predict
```

Compute accuracy:

```shell
$ 
```
Build dataset:
$ python build_dataset.py
```

Compute accuracy:
```shell
$ python accuracies.py ourmodel path_to_predictions.csv[default=predictions/CNN_dev_predictions.csv] path_to_labels.npy[default=datasets/processed/dev_labels.npy]
```