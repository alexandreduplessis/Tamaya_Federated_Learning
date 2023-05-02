import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


DATAFOLDER = './datasets/' # Folder where to store datasets.

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, train=True, device='cpu'):
        super().__init__()

        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


        data = datasets.CIFAR10(DATAFOLDER, train=train, download=True)
        self.targets = data.targets
        self.data = torch.stack([transform(img) for img in data.data]).to(dtype=torch.float32, device=device)
        self.labels = torch.tensor(data.targets, dtype=torch.int64, device=device)
        self.size = len(data.targets)
        self.transforms = None # transforms.RandomHorizontalFlip(p=0.5) à implémenter

    def __len__(self): return self.size
    def __getitem__(self, index): return (self.data[index], self.labels[index])

class FMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, device='cpu'):
        super().__init__()

        transform = None
        data = datasets.FashionMNIST(DATAFOLDER, train=train, download=True)
        self.targets = data.targets
        self.data = data.data.clone().detach().to(dtype=torch.float32, device=device)
        self.data.unsqueeze_(1)
        self.labels = data.targets.clone().detach().to(dtype=torch.int64, device=device)
        self.size = len(data.targets)
        self.transforms = None

    def __len__(self): return self.size
    def __getitem__(self, index): return (self.data[index], self.labels[index])

class MNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, device='cpu'):
        super().__init__()

        transform = None
        data = datasets.MNIST(DATAFOLDER, train=train, download=True)
        self.targets = data.targets
        self.data = data.data.clone().detach().to(dtype=torch.float32, device=device)
        self.data.unsqueeze_(1)
        self.labels = data.targets.clone().detach().to(dtype=torch.int64, device=device)
        self.size = len(data.targets)
        self.transforms = None

    def __len__(self): return self.size
    def __getitem__(self, index): return (self.data[index], self.labels[index])

class SVHN(torch.utils.data.Dataset):
    def __init__(self, train=True, device='cpu'):
        super().__init__()

        transform = None
        data = datasets.SVHN(DATAFOLDER, split='train' if train else 'test', download=True)
        self.targets = data.labels
        self.data = torch.tensor(data.data, dtype=torch.float32, device=device)
        self.labels = torch.tensor(data.labels, dtype=torch.int64, device=device)
        self.size = len(data.labels)
        self.transforms = None

    def __len__(self): return self.size
    def __getitem__(self, index): return (self.data[index], self.labels[index])