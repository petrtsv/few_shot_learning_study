import torch
from torch.utils.data import Dataset

from data.base import LabeledSubdataset


class EpisodeSampler(Dataset):
    def __init__(self, subdataset: LabeledSubdataset, n_way: int, n_shot: int, batch_size: int, balanced: bool,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.subdataset = subdataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.device = device
        self.balanced = balanced

    def sample(self):
        cur_subdataset, _ = self.subdataset.extract_classes(self.n_way)
        support_subdataset, query_subdataset = cur_subdataset.extract_balanced(self.n_shot)
        classes_mapping = {}

        support_set_labels = support_subdataset.labels()

        support_set = [[] for i in range(len(support_set_labels))]

        h = 0
        for label in support_set_labels:
            classes_mapping[label] = h
            h += 1

        for i in range(len(support_subdataset)):
            item, label, _ = support_subdataset[i]
            support_set[classes_mapping[label]].append(item.to(self.device))
        if not self.balanced:
            batch = query_subdataset.random_batch(self.batch_size)
        else:
            batch = query_subdataset.balanced_batch(self.batch_size)

        for i in range(len(batch[1])):
            batch[1][i] = classes_mapping[batch[1][i].item()]

        for i in range(len(support_set)):
            support_set[i] = torch.stack(support_set[i])
        support_set = torch.stack(support_set)

        return support_set, batch


class EpisodeSamplerGlobalLabels(EpisodeSampler):
    def sample(self):
        cur_subdataset, _ = self.subdataset.extract_classes(self.n_way)
        support_subdataset, query_dataset = cur_subdataset.extract_balanced(self.n_shot)
        classes_mapping = {}

        support_set_labels = support_subdataset.labels()

        support_set = [[] for i in range(len(support_set_labels))]

        h = 0
        for label in support_set_labels:
            classes_mapping[label] = h
            h += 1

        for i in range(len(support_subdataset)):
            item, label, _ = support_subdataset[i]
            support_set[classes_mapping[label]].append(item.to(self.device))
        if not self.balanced:
            batch = list(query_dataset.random_batch(self.batch_size))
        else:
            batch = list(query_dataset.balanced_batch(self.batch_size))
        # batch = list(cur_subdataset.random_batch(self.batch_size))
        # batch.append([-1] * len(batch[1]))
        for i in range(len(batch[1])):
            # batch[2][i] = batch[1][i].item()
            batch[1][i] = classes_mapping[batch[1][i].item()]
        # batch[2] = torch.tensor(batch[2])

        for i in range(len(support_set)):
            support_set[i] = torch.stack(support_set[i])
        support_set = torch.stack(support_set)

        return support_set, batch, classes_mapping
