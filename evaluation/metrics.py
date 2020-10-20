import torch
from sklearn.metrics import accuracy_score


def accuracy(predictions: torch.Tensor, target: torch.Tensor):
    labels_pred = predictions.argmax(dim=1)
    return accuracy_score(labels_pred.cpu(), target.cpu())


def top_k_accuracy(predictions: torch.Tensor, target: torch.Tensor, k: int):
    labels_pred = torch.argsort(predictions, dim=1, descending=True)[:, :k]
    score = 0
    number = labels_pred.size(0)

    i: int = 0
    for row in labels_pred:
        if target[i] in row:
            score += 1
        i += 1

    return score / number
