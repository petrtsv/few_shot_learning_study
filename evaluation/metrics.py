import torch
from sklearn.metrics import accuracy_score


def accuracy(predictions: torch.Tensor, target: torch.Tensor):
    labels_pred = predictions.argmax(dim=1)
    return accuracy_score(labels_pred.cpu(), target.cpu())
