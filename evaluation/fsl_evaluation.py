import time

import torch

from data import LABELED_DATASETS
from data.samplers import EpisodeSampler
from evaluation.metrics import accuracy
from experiments_index.index import save_record
from models.base_model import FSLSolver
from utils import pretty_time


def evaluate_fsl_solution(model: FSLSolver, sampler: EpisodeSampler, n_iterations: int, metrics_prefix: str = ""):
    device = sampler.device
    accuracy_sum = 0
    model.eval()
    with torch.no_grad():
        for iteration in range(n_iterations):
            support_set, batch = sampler.sample()
            support_set = support_set.to(device)
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            prototypes = model.get_prototypes(support_set)
            y_pred = model.inference(prototypes, x)
            cur_accuracy = accuracy(predictions=y_pred, target=y)
            accuracy_sum += cur_accuracy

    return {metrics_prefix + 'accuracy': accuracy_sum / n_iterations}


def test_model(model: FSLSolver, options: dict, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("Testing started for parameters:")
    print(options)
    print()
    start_time = time.time()

    dataset = LABELED_DATASETS[options['test_dataset']](image_size=options['image_size'])
    subdataset = dataset.subdataset
    subdataset.set_test(True)

    sampler = EpisodeSampler(subdataset=subdataset, n_way=options['test_n_way'], n_shot=options['n_shot'],
                             batch_size=options['test_batch_size'], balanced=False, device=device)

    results = evaluate_fsl_solution(model, sampler, options['test_n_iterations'], metrics_prefix="test_")

    cur_time = time.time()
    testing_time = cur_time - start_time
    results['testing_time'] = testing_time
    save_record("'Few-Shot Learning testing", {**options, **results})
    print("Testing finished. Total execution time: %s" % pretty_time(testing_time))
    print("Results:", results)
    print()
