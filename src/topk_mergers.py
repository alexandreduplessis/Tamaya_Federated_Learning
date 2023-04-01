import torch
import heapq
from collections import OrderedDict

class Merger_FedTopK:
    def __init__(self, k):
        """
        k is either a float or a list:
        |k| is the ratio of clients we take at each epoch.
        - k > 0 => FedMax(k)
        - k < 0 => FedMin(k)
        """
        self.k = k

    def __call__(self, outputs):
        names = outputs[0].weight.keys()

        k = self.k[outputs[0].round] if isinstance(self.k, list) else self.k
        nk = max(1, int(abs(k)*len(outputs)))

        losses = {output.client_id: output.losses[-1] for output in outputs}
        if k < 0: top_k = heapq.nsmallest(nk, losses, key=lambda cid: losses[cid])
        else: top_k = heapq.nlargest(nk, losses, key=lambda cid: losses[cid])

        alphas = [1./nk if (output.client_id in top_k) else 0.0 for output in outputs]
        return OrderedDict([(name, torch.sum(torch.stack([output.weight[name]*(1./nk if (output.client_id in top_k) else 0.0) for output in outputs]), dim=0)) for name in names]), alphas

    def reset(self):
        return self