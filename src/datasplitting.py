import torch
import random
import math
import numpy as np


def split_dataset(dataset, nb_clients, shard_size=1000):
    """
    1. Sorts the `dataset` into small chunks of size `shard_size`.
    2. Distributes these chunks into `nb_clients` clients.
    3. a) `shard_size` needs to be small enough.
       b) each client gets at least one chunk.
    """
    subsets = [[] for n in range(nb_clients)]
    items = list(sorted(range(len(dataset)), key=lambda i: dataset.targets[i] + random.uniform(0,1)))
    shards = [items[shard_size*i:shard_size*i+shard_size] for i in range(math.ceil(len(dataset)/shard_size))]
    while shards:
        if any(not(subset) for subset in subsets):
            n = random.choice([i for i in range(nb_clients) if not(subsets[i])])
            subsets[n].extend(shards.pop())
        else:
            subsets[random.randrange(nb_clients)].extend(shards.pop())
    return [torch.utils.data.Subset(dataset, subset) for subset in subsets]

def split_dataset_iid(dataset, nb_clients):
    """
    1. Distrbutes the `dataset` evenly between `nb_clients`.
    """
    items = list(range(len(dataset)))
    np.random.shuffle(items)
    size = math.ceil(len(dataset) / nb_clients)
    subsets =  [items[size*i:size*(i+1)] for i in range(nb_clients)]
    return [torch.utils.data.Subset(dataset, subset) for subset in subsets]