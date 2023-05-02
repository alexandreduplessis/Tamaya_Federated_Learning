import torch
from collections import OrderedDict

class Merger_FedAvg:
    def __call__(self, outputs, accs_list):
        names = outputs[0].weight.keys()
        sum_sizes = sum([output.size for output in outputs])
        return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)) for name in names]), [output.size/sum_sizes for output in outputs]

    def reset(self):
        return self
    
def partial_merging_i(name, outputs, sum_sizes, clients_list):
    nb_clients = len(clients_list[name])
    return torch.sum(torch.stack([outputs[clients_list[name][j]].weight[name]/nb_clients for j in range(nb_clients)]), dim=0)

class Merger_Layer:
    def __call__(self, outputs, i):
        names = outputs[0].weight.keys()
        sum_sizes = sum([output.size for output in outputs])
        nb_layers = len(outputs[0].weight)
        # divide the set of clients into nb_layers random subsets (but with no empty subset)
        clients_list = {name: [] for name in names}
        for client in range(len(outputs)):
            # choose a random layer in names
            layer = torch.randint(0, nb_layers, (1,)).item()
            layer_name = list(names)[layer]
            # add the client to the list of clients for this layer
            clients_list[layer_name].append(client)
        # check that no subset is empty
        for name in names:
            print(len(clients_list[name]))
            if len(clients_list[name]) == 0:
                # if a subset is empty, we add a random client to it
                clients_list[name].append(torch.randint(0, len(outputs), (1,)).item())
        # merge the weights
        return OrderedDict([(name, partial_merging_i(name, outputs, sum_sizes,  clients_list)) for name in names]), [output.size/sum_sizes for output in outputs]

    def reset(self):
        return self