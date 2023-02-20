import torch
import numpy as np
from collections import OrderedDict

class Merger_FedSoft:
    def __init__(self, T):
        """
        1. T is either a float or a list
        - T[t] > 0 => FedSoftmax
        - T[t] = 0 => FedCst
        - T[t] < 0 => FedSoftmin
        """
        self.T = T

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        T = self.T[outputs[0].round] if isinstance(self.T, list) else self.T

        alpha = {output.client_id: np.exp(T * output.losses[-1]) for output in outputs}
        sum_alpha = sum([alpha[output.client_id] for output in outputs])

        return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names])

    def reset(self):
        return self

class Merger_FedSuperSoft:
    def __init__(self, T):
        """
        1. T is either a float or a list
        - T[t] > 0 => FedSuperSoftmax
        - T[t] = 0 => FedAvg
        - T[t] < 0 => FedSuperSoftmin
        """
        self.T = T

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        T = self.T[outputs[0].round] if isinstance(self.T, list) else self.T

        alpha = {output.client_id: output.size * np.exp(T * output.losses[-1]) for output in outputs}
        sum_alpha = sum([alpha[output.client_id] for output in outputs])

        return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names])

    def reset(self):
        return self