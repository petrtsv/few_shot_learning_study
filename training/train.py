import copy
import json
import os
import time
from uuid import uuid4

import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

from config import EXPERIMENTS_DIR
from data import LABELED_DATASETS
from data.base import LabeledSubdataset
from data.samplers import EpisodeSampler, EpisodeSamplerGlobalLabels
from evaluation.fsl_evaluation import evaluate_fsl_solution
from experiments_index.index import save_record
from models.base_model import FSLSolver
from models.feature_extarctors import FEATURE_EXTRACTORS
from utils import pretty_time
from visualization.plots import PlotterWindow


def lr_schedule(iteration: int):
    if iteration >= 30000:
        return 0.0012
    elif iteration >= 20000:
        return 0.006
    else:
        return 0.1


def run_training(model: FSLSolver, train_subdataset: LabeledSubdataset,
                 val_subdataset: LabeledSubdataset, options: dict,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model = model.to(device)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, nesterov=True, weight_decay=0.0005,
                                momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    train_sampler = EpisodeSamplerGlobalLabels(subdataset=train_subdataset, n_way=options['n_way'],
                                               n_shot=options['n_shot'],
                                               batch_size=options['batch_size'], balanced=True, device=device)
    val_sampler = EpisodeSampler(subdataset=val_subdataset, n_way=options['n_way'], n_shot=options['n_shot'],
                                 batch_size=options['batch_size'], balanced=True, device=device)

    best_model = copy.deepcopy(model)
    best_accuracy = 0
    best_accuracy_metrics = None
    best_iteration = -1

    losses_plotter = PlotterWindow(interval=1000)
    metrics_plotter = PlotterWindow(interval=1000)

    print("Training started for parameters:")
    print(options)
    print()

    start_time = time.time()

    train_losses_list = []
    train_metrics_list = []
    test_metrics_list = []

    for iteration in range(options['n_iterations']):
        model.train()

        support_set, batch, labels_mapping = train_sampler.sample()
        support_set = support_set.to(device)

        query_set, query_labels = batch
        query_set = query_set.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad()
        loss, losses, metrics = model(support_set, query_set, query_labels, labels_mapping)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses_list.append(losses)
        train_metrics_list.append(metrics)

        for loss_item in losses.keys():
            losses_plotter.add_point("train:" + loss_item, iteration, losses[loss_item])

        for metric_item in metrics.keys():
            metrics_plotter.add_point("train:" + metric_item, iteration, metrics[metric_item])

        if iteration % options['eval_period'] == 0 or iteration == options['n_iterations'] - 1:
            val_start_time = time.time()

            val_metrics = evaluate_fsl_solution(model, val_sampler, options['eval_iterations'])

            for metric_item in val_metrics.keys():
                metrics_plotter.add_point("test:" + metric_item, iteration, val_metrics[metric_item])

            val_metrics['iteration'] = iteration

            cur_time = time.time()

            val_time = cur_time - val_start_time
            time_used = cur_time - start_time
            time_per_iteration = time_used / (iteration + 1)

            val_metrics['time'] = val_time

            test_metrics_list.append(val_metrics)

            print()
            print("[%d/%d] = %.2f%%\t\tLoss: %.4f" % (
                iteration + 1, options['n_iterations'], (iteration + 1) / options['n_iterations'] * 100, loss.item()))
            print("Current validation time: %s" % pretty_time(val_time))

            print('Average iteration time: %s\tEstimated execution time: %s' % (
                pretty_time(time_per_iteration),
                pretty_time(time_per_iteration * (options['n_iterations'] - iteration - 1)),
            ))
            print("Current metrics:", val_metrics)
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                best_accuracy_metrics = val_metrics
                best_iteration = iteration
                best_model = copy.deepcopy(model)
                print("Best evaluation result yet!")
            print()

    cur_time = time.time()
    training_time = cur_time - start_time

    options['time'] = training_time
    for metric_item in best_accuracy_metrics.keys():
        options[metric_item] = best_accuracy_metrics[metric_item]

    result_directory = os.path.join(EXPERIMENTS_DIR, str(uuid4()))
    os.makedirs(result_directory, exist_ok=True)
    options['result_directory'] = result_directory

    torch.save(best_model, os.path.join(result_directory, 'best_model.pt'))

    if device.type == 'cuda':
        scripted_best_model_gpu = torch.jit.script(best_model)
        torch.jit.save(scripted_best_model_gpu, os.path.join(result_directory, 'scripted_best_model_gpu.pts'))

    scripted_best_model_cpu = torch.jit.script(best_model.cpu())
    torch.jit.save(scripted_best_model_cpu, os.path.join(result_directory, 'scripted_best_model_cpu.pts'))

    train_losses_df = pd.DataFrame.from_records(train_losses_list)

    train_metrics_df = pd.DataFrame.from_records(train_metrics_list)
    test_metrics_df = pd.DataFrame.from_records(test_metrics_list)

    train_losses_df.to_csv(os.path.join(result_directory, 'train_losses.csv'))
    train_metrics_df.to_csv(os.path.join(result_directory, 'train_metrics.csv'))
    test_metrics_df.to_csv(os.path.join(result_directory, 'test_metrics.csv'))

    with open(os.path.join(result_directory, 'options.json'), 'w') as f:
        print(json.dumps(options), file=f)

    save_record('Few-Shot Learning training', options)

    print("Training finished. Total execution time: %s" % pretty_time(training_time))
    print("Best validation accuracy is: %.3f" % best_accuracy)
    print("Best iteration is: [%d/%d]" % (best_iteration + 1, options['n_iterations']))
    print()

    return best_model.to(device)


def train_model(options: dict):
    dataset = LABELED_DATASETS[options['dataset']](image_size=options['image_size'])
    train_subdataset, test_subdataset = dataset.subdataset.extract_classes(options['train_classes'])

    train_subdataset.set_test(False)
    test_subdataset.set_test(True)

    model = FSLSolver(
        backbone=FEATURE_EXTRACTORS[options['backbone']],
        aux_rotation_k=options['aux_rotation_k'],
        aux_location_k=options['aux_location_k'],
        dfmn_k=options['dfmn_k'],
        train_n_way=options['n_way'],
        dataset_classes=dataset.CLASSES,
        k=options['k'],
    )

    trained_model = run_training(model=model, train_subdataset=train_subdataset, val_subdataset=test_subdataset,
                                 options=options)

    return trained_model, options
