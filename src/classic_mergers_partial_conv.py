import torch
from collections import OrderedDict

def partial_merging_conv(name, outputs, sum_sizes, i):
    if name[0] == 'c':
        return torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)
    else:
        return outputs[i].weight[name]

class Merger_FedParConv:
    def __call__(self, outputs, i):
        names = outputs[0].weight.keys()
        sum_sizes = sum([output.size for output in outputs])
        return OrderedDict([(name, partial_merging_conv(name, outputs, sum_sizes, i)) for name in names])

    def reset(self):
        return self