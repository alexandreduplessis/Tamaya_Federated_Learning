import torch
import random
import math
import numpy as np


def split_dataset_noniid_old(dataset, nb_clients, shard_size=1000):
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


def split_dataset_iid_old(dataset, nb_clients):
    """
    1. Distrbutes the `dataset` evenly between `nb_clients`.
    """
    items = list(range(len(dataset)))
    np.random.shuffle(items)
    size = math.ceil(len(dataset) / nb_clients)
    subsets =  [items[size*i:size*(i+1)] for i in range(nb_clients)]
    return [torch.utils.data.Subset(dataset, subset) for subset in subsets]


def split_dataset_iid(dataset, nb_clients, ratio_test=0.2):
    assert(len(dataset) > nb_clients)
    items = list(range(len(dataset)))
    np.random.shuffle(items)
    size = math.ceil(len(dataset) / nb_clients)
    subsets = [items[size*i:size*(i+1)] for i in range(nb_clients)]

    if ratio_test == 0:
        return [{'train': torch.utils.data.Subset(dataset, subset),
                 'test' : None} for subset in subsets]
    return [{'train': torch.utils.data.Subset(dataset, subset[:-int(ratio_test * len(subset))]),
             'test' : torch.utils.data.Subset(dataset, subset[-int(ratio_test * len(subset)):])}
            for subset in subsets]


def split_dataset_noniid(dataset, nb_clients, size_list, numbers_list, ratio_test=0.2):
    # numbers_list is a list of pourcentages of each number for each client
    # size_list is a list of the size of each client
    assert(len(dataset) > nb_clients)
    result = []
    # separate the dataset into 10 subsets, one for each number
    subsets = [[] for n in range(10)]
    items = list(sorted(range(len(dataset)), key=lambda i: dataset.targets[i] + random.uniform(0,1)))
    for item in items:
        subsets[dataset.targets[item]].append(item)
    for client_id in range(nb_clients):
        assert(len(numbers_list[client_id]) == 10)
        assert(1 - sum(numbers_list[client_id]) < 0.1)
        # select the number of items for each client
        client_subset_train = []
        client_subset_test = []
        for number in range(10):
            number_size = int(numbers_list[client_id][number] * size_list[client_id])
            # randomly select the items
            np.random.shuffle(subsets[number])
            client_subset_train.extend(torch.utils.data.Subset(dataset, subsets[number][:number_size - int(ratio_test * number_size)]))
            client_subset_test.extend(torch.utils.data.Subset(dataset, subsets[number][number_size - int(ratio_test * number_size):number_size]))
        result.append({'train': client_subset_train, 'test': client_subset_test})
    return result