
import torch

from src.convnet import ConvNet
from src.tf_cifar import TFCifar
from src.datasets import MNIST, FMNIST, CIFAR10
from src.accuracy import get_accuracy

if __name__ == "__main__":
    device = 'cuda'
    epochs = 100
    batchsize = 64

    data_train = MNIST(True, device)
    data_test = MNIST(False, device)
    model = ConvNet()
    model.to(device)

    accs = []

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    testloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=batchsize, shuffle=True, num_workers=0)
    optim = torch.optim.SGD(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        model.train()
        for (x, y) in dataloader:
            optim.zero_grad()
            y_out = model(x)
            loss = loss_fn(y_out, y)
            loss.backward()
            optim.step()
        accs.append(get_accuracy(model, testloader))
        print(f"{epoch+1}/{epochs}: {accs[-1]}")
    print(accs)

