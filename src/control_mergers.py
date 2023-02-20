import torch
from collections import OrderedDict


class Merger_FedWCostAvg:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_losses = None

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        sum_sizes = sum([output.size for output in outputs])
        if self.last_losses is None:
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)) for name in names])
        else:
            sum_D = sum([self.last_losses[output.client_id]/output.losses[-1] for output in outputs])
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(self.alpha*output.size/sum_sizes + (1-self.alpha)*(self.last_losses[output.client_id]/output.losses[-1])/sum_D) for output in outputs]), dim=0)) for name in names])
        self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
        return res

    def reset(self):
        self.last_losses = None
        return self

class Merger_FedDiff:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_losses = None

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        sum_sizes = sum([output.size for output in outputs])
        if self.last_losses is None:
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)) for name in names])
        else:
            sum_D = sum([output.losses[-1]-self.last_losses[output.client_id] for output in outputs])
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(self.alpha*output.size/sum_sizes + (1-self.alpha)*(output.losses[-1]-self.last_losses[output.client_id])/sum_D) for output in outputs]), dim=0)) for name in names])
        self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
        return res

    def reset(self):
        self.last_losses = None
        return self

class Merger_FedControl1:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.last_losses = None
        self.sum_losses = None

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        sum_sizes = sum([output.size for output in outputs])
        if self.last_losses is None:
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)) for name in names])
            self.sum_losses = {output.client_id: 0.1 for output in outputs}
        else:
            sum_D = sum([self.last_losses[output.client_id]/output.losses[-1] for output in outputs])
            sum_I = sum([self.sum_losses[output.client_id] for output in outputs])
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(self.alpha*output.size/sum_sizes + self.beta*(self.last_losses[output.client_id]/output.losses[-1])/sum_D + (1-self.alpha-self.beta)*(self.sum_losses[output.client_id]/sum_I)) for output in outputs]), dim=0)) for name in names])
        self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
        self.sum_losses = {output.client_id: self.sum_losses[output.client_id] + output.losses[-1] for output in outputs}
        return res

    def reset(self):
        self.last_losses = None
        self.sum_losses = None
        return self

class Merger_FedControl2:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.last_losses = None
        self.sum_losses = None

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        sum_sizes = sum([output.size for output in outputs])
        if self.last_losses is None:
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*output.size/sum_sizes for output in outputs]), dim=0)) for name in names])
            self.sum_losses = {output.client_id: 0.1 for output in outputs}
        else:
            sum_D = sum([self.last_losses[output.client_id]/output.losses[-1] for output in outputs])
            sum_I = sum([self.sum_losses[output.client_id] for output in outputs])
            res = OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(self.alpha*output.size/sum_sizes + self.beta*(self.last_losses[output.client_id]/output.losses[-1])/sum_D + (1-self.alpha-self.beta)*(self.sum_losses[output.client_id]/sum_I)) for output in outputs]), dim=0)) for name in names])
        self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
        self.sum_losses = {output.client_id: 0.8 * self.sum_losses[output.client_id] + output.losses[-1] for output in outputs}
        return res

    def reset(self):
        self.last_losses = None
        self.sum_losses = None
        return self
