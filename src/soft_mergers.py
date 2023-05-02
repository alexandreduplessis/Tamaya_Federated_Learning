import torch
import numpy as np
from collections import OrderedDict

class Merger_FedSoft:
    def __init__(self, T, proportional=False):
        """
        1. T is either a float or a list
        - T[t] > 0 => FedSoftmax
        - T[t] = 0 => FedCst
        - T[t] < 0 => FedSoftmin
        """
        self.T = T
        self.proportional = proportional

    def __call__(self, outputs, accs_list):
        names = outputs[0].weight.keys()

        T = self.T[outputs[0].round] if isinstance(self.T, list) else self.T

        if not self.proportional:
            alpha = {output.client_id: np.exp(T * output.losses[-1]) for output in outputs}
            sum_alpha = sum([alpha[output.client_id] for output in outputs])

            return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
        
        else:
            # weight each client proportionally to its loss
            alpha = {output.client_id: output.losses[-1] for output in outputs}
            sum_alpha = sum([alpha[output.client_id] for output in outputs])

            return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]

    def reset(self):
        return self


class Merger_FedSoftTop:
    def __init__(self, T, proportional=False):
        """
        1. T is either a float or a list
        - T[t] > 0 => FedSoftmax
        - T[t] = 0 => FedCst
        - T[t] < 0 => FedSoftmin
        """
        self.T = T
        self.proportional = proportional
        self.loss_history = None

    def __call__(self, outputs, accs_list):
        names = outputs[0].weight.keys()

        T = self.T[outputs[0].round] if isinstance(self.T, list) else self.T
        if self.loss_history is None:
            if not self.proportional:
                alpha = {output.client_id: np.exp(T * output.losses[-1]) for output in outputs}
                self.loss_history = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
            
            else:
                # weight each client proportionally to its loss
                alpha = {output.client_id: output.losses[-1] for output in outputs}
                self.loss_history = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
        else:
            if not self.proportional:
                alpha = {output.client_id: np.exp(T * np.absolute(output.losses[-1] - self.loss_history[output.client_id]))*output.losses[-1] for output in outputs}
                self.loss_history = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
            
            else:
                # weight each client proportionally to its loss
                alpha = {output.client_id: np.absolute(output.losses[-1] - self.loss_history[output.client_id])*output.losses[-1] for output in outputs}
                self.loss_history = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]

    def reset(self):
        return self

class Merger_FedHess:
    def __init__(self, T, proportional=False):
        """
        1. T is either a float or a list
        - T[t] > 0 => FedSoftmax
        - T[t] = 0 => FedCst
        - T[t] < 0 => FedSoftmin
        """
        self.T = T
        self.proportional = proportional
        self.last_losses = None

    def __call__(self, outputs, accs_list):
        names = outputs[0].weight.keys()

        T = self.T[outputs[0].round] if isinstance(self.T, list) else self.T
        if self.last_losses is None:
            if not self.proportional:
                alpha = {output.client_id: np.exp(T * output.losses[-1]) for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
            
            else:
                # weight each client proportionally to its loss
                alpha = {output.client_id: output.losses[-1] for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
        else:
            if not self.proportional:
                alpha = {output.client_id: np.exp(T * (output.losses[-1] - self.last_losses[output.client_id])) for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
            
            else:
                # weight each client proportionally to its loss
                alpha = {output.client_id: (output.losses[-1] - self.last_losses[output.client_id]) for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]

    def reset(self):
        return self

class Merger_FednegHess:
    def __init__(self, T, proportional=False):
        """
        1. T is either a float or a list
        - T[t] > 0 => FedSoftmax
        - T[t] = 0 => FedCst
        - T[t] < 0 => FedSoftmin
        """
        self.T = T
        self.proportional = proportional
        self.last_losses = None

    def __call__(self, outputs, accs_list):
        names = outputs[0].weight.keys()

        T = self.T[outputs[0].round] if isinstance(self.T, list) else self.T
        if self.last_losses is None:
            if not self.proportional:
                alpha = {output.client_id: np.exp(T * output.losses[-1]) for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
            
            else:
                # weight each client proportionally to its loss
                alpha = {output.client_id: output.losses[-1] for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
        else:
            if not self.proportional:
                alpha = {output.client_id: np.exp(-T * (output.losses[-1] - self.last_losses[output.client_id])) for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]
            
            else:
                # weight each client proportionally to its loss
                alpha = {output.client_id: -(output.losses[-1] - self.last_losses[output.client_id]) for output in outputs}
                self.last_losses = {output.client_id: output.losses[-1] for output in outputs}
                sum_alpha = sum([alpha[output.client_id] for output in outputs])

                return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(alpha[output.client_id]/sum_alpha) for output in outputs]), dim=0)) for name in names]), [alpha[output.client_id]/sum_alpha for output in outputs]

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