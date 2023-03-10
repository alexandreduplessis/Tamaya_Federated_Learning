from src.datasets import MNIST
from src.datasplitting import *


if __name__ == "__main__":
    device = 'cuda'
    data_train = MNIST(True, device)
    for cid, client in enumerate(split_dataset_noniid(data_train, 20, ratio_test=0.15)):
        print(cid, len(client['train']), client['test'])