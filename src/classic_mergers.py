import torch
from collections import OrderedDict

class Merger_FedAvg:
    def __call__(self, outputs):
        names = outputs[0].weight.keys()
        sum_sizes = sum([output.size for output in outputs])
        alpha = [output.size / sum_sizes for output in outputs]
        return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*alpha[o] for o, output in enumerate(outputs)]), dim=0)) for name in names]), alpha

    def reset(self):
        return self