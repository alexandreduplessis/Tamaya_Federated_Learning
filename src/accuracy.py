import torch


def get_accuracy(model, dl):
    """
    1. Computes the accuracy of `model` using `dataloader`.
    """
    model.eval()
    success, total = 0.0, 0
    for (x, y) in dl:
        with torch.no_grad():
            y_out = model(x)
            success += torch.eq(torch.argmax(y_out, dim=1), y).float().sum().item()
            total += y.shape[0]

    return 100 * success / total