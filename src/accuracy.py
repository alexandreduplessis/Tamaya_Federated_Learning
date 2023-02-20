import torch


def get_accuracy(model, dataloader):
    """
    1. Computes the accuracy of `model` using `dataloader`.
    """
    model.eval()
    success, total = 0.0, 0
    for (x, y) in dataloader:
        with torch.no_grad():
            y_out = model(x)
            success += (torch.argmax(y_out, dim=1) == y).sum().item()
            total += y.shape[0]
    return success / total