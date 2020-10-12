import torch

from data.samplers import EpisodeSampler
from evaluation.metrics import accuracy
from models.base_model import FSLSolver


def evaluate_fsl_solution(model: FSLSolver, sampler: EpisodeSampler, n_iterations: int):
    device = sampler.device
    accuracy_sum = 0
    model.eval()
    with torch.no_grad():
        for iteration in range(n_iterations):
            support_set, batch = sampler.sample()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            prototypes = model.get_prototypes(support_set)
            y_pred = model.inference(prototypes, x)
            cur_accuracy = accuracy(predictions=y_pred, target=y)
            accuracy_sum += cur_accuracy

    return {'accuracy': accuracy_sum / n_iterations}
