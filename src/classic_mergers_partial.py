import torch
from collections import OrderedDict

def partial_merging(name, outputs, sum_sizes, i):
    if name[0] == 'l' and name[0:4] != 'lin1':
        return torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)
    else:
        return outputs[i].weight[name]

class Merger_FedPar:
    def __call__(self, outputs, i):
        names = outputs[0].weight.keys()
        sum_sizes = sum([output.size for output in outputs])
        return OrderedDict([(name, partial_merging(name, outputs, sum_sizes, i)) for name in names])

    def reset(self):
        return self