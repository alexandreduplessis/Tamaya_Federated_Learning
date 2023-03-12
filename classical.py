
import torch

from src.convnet import ConvNet
from src.tf_cifar import TFCifar
from src.datasets import MNIST, FMNIST, CIFAR10
from src.accuracy import get_accuracy

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = 'cuda'
    epochs = 100
    batchsize = 64

    data_train = FMNIST(True, device)
    data_test = FMNIST(False, device)
    model = ConvNet()
    model.to(device)

    accs = []

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=batchsize, shuffle=True, num_workers=0)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    tt = []

    for epoch in range(epochs):
        model.train()

        losses = []
        for (x, y) in dataloader:
            optim.zero_grad()
            y_out = model(x)
            loss = loss_fn(y_out, y)
            with torch.no_grad():
                losses.append(loss.detach().cpu().item())
            loss.backward()
            optim.step()
        print(sum(losses)/len(losses))
        tt.append(sum(losses)/len(losses))

        testloader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=False, num_workers=0)
        accs.append(get_accuracy(model, testloader))
        print(f"{epoch+1}/{epochs}: batchsize=100: {accs[-1]}")


    plt.plot(tt)
    plt.show()

