# Federated Learning Aggregation: New Robust Algorithm with Guarantees
This repository contains our empirical results for multiple federated learning aggregation strategies. Our code uses Pytorch and sequentially computes the federated learning algorithm with several formulas discussed in the article [cite our article].

### Datasets used
- MNIST
- FashionMNIST (or FMNIST for short)
- CIFAR-10

### Models on each dataset
- ConvNet
- ConvNet
- TF_Cifar

### Structure of the repository
- `train.py` runs the federated learning algorithm and adds curves to the `outputs` directory as `{filename}_accs.inf` for accuracies and `{output}_loss.inf` for losses. It takes multiple parameters such as `dataset` (MNIST, FMNIST & CIFAR10), `clients` (number of clients), `experiment` (a named set of strategies that have to be computed), `output` (name of output file), ... Run `python3 train.py --help` to learn more about it or check the code.
- `display.py {filename} {option1} {option2}` plots the curves contained in `outputs\{filename}.crv` with options specified as arguments. First option adds confidence interval and the second one plots each curve (not only the average of multiple curves) if set to 1.
- The folder `src` contains: `datasets.py` a file that loads any of our datasets on the proper device; `datasplitting.py` a script that splits a dataset into multiple clients with a non-IID distribution; `X_mergers.py` files where our aggregation formulas are written; it also contains our models and some useful functions.
