from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.metrics import accuracy
from models.auxillary_tasks import RotationTask
from models.dfmn import DFMNLoss
from models.feature_extarctors.base import NoFlatteningBackbone
from utils import remove_dim

MAX_BATCH_SIZE = 500


class FSLSolver(nn.Module):
    def __init__(self, backbone: NoFlatteningBackbone, aux_rotation_k=0.0, aux_location_k=0.0, dfmn_k=0.0,
                 train_classes=None, train_n_way=None):
        super(FSLSolver, self).__init__()
        self.feature_extractor = backbone

        self.train_classes = train_classes
        self.train_n_way = train_n_way

        self.aux_rotation_k = aux_rotation_k
        self.aux_rotation_task = None

        self.aux_location_k = aux_location_k
        self.aux_location_task = None

        self.dfmn_k = dfmn_k
        self.dfmn_loss = None

        if self.aux_rotation_k > 10 ** -9:
            self.aux_rotation_task = RotationTask(self.feature_extractor.output_features(),
                                                  self.feature_extractor.output_featmap_size())

        if self.dfmn_k > 10 ** -9:
            self.dfmn_loss = DFMNLoss(self.feature_extractor.output_features(),
                                      self.feature_extractor.output_featmap_size(), self.train_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                try:
                    nn.init.constant_(m.bias, 0)
                except AttributeError:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, batch: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        # minibatches = batch.split(split_size=MAX_BATCH_SIZE)
        # xs = []
        # for minibatch in minibatches:
        #     output = self.feature_extractor(minibatch)
        #     output = output.view(output.size(0), -1)
        #     xs.append(output)
        # x = torch.cat(xs)
        features = self.feature_extractor(batch)

        if flatten:
            return features.view(features.size(0), -1)
        else:
            return features

    def compute_prototypes(self, support_set: torch.Tensor):
        return torch.mean(support_set, dim=1)

    @torch.jit.export
    def get_prototypes(self, support_set: torch.Tensor):
        n_classes = support_set.size(0)
        support_set_size = support_set.size(1)
        support_set_features = self.extract_features(remove_dim(support_set, 1)).view(n_classes,
                                                                                      support_set_size, -1)
        class_prototypes = self.compute_prototypes(support_set_features)
        return class_prototypes

    def scores(self, prototypes: torch.Tensor, query_set_features: torch.Tensor,
               is_train: bool = False) -> torch.Tensor:
        n_classes = prototypes.size(0)
        query_set_size = query_set_features.size(0)

        class_prototypes = prototypes

        query_set_features_prepared = query_set_features.unsqueeze(1).repeat_interleave(repeats=n_classes,
                                                                                        dim=1)

        distance = torch.sum((class_prototypes.unsqueeze(0).repeat_interleave(repeats=query_set_size,
                                                                              dim=0) -
                              query_set_features_prepared).pow(2), dim=2)

        return -distance

    @torch.jit.ignore
    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor, labels: torch.Tensor,
                global_classes_mapping: dict) -> Tuple[torch.Tensor, dict, dict]:
        combined_input = torch.cat((remove_dim(support_set, 1), query_set), 0)

        query_set_features = self.extract_features(query_set)

        prototypes = self.get_prototypes(support_set)

        scores = self.scores(prototypes=prototypes, query_set_features=query_set_features, is_train=True)

        loss = self.loss_fn(scores, labels)
        losses = {'fsl_loss': loss.item()}

        if self.aux_rotation_task is not None:
            rotated, rotation_labels = self.aux_rotation_task.get_task(combined_input)
            rotated_features = self.extract_features(rotated, flatten=False)
            rot_loss = self.aux_rotation_task(rotated_features, rotation_labels)
            losses['rotation_loss'] = rot_loss.item()
            loss += rot_loss

        if self.dfmn_loss is not None:
            loss_dfmn = self.dfmn_loss(query_set_features, labels, global_classes_mapping, self.train_n_way)
            losses['dfmn_loss'] = loss_dfmn.item()
            loss += loss_dfmn

        metrics = {'accuracy': accuracy(scores, labels)}

        return loss, losses, metrics

    @torch.jit.export
    def inference(self, prototypes: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.scores(prototypes, query_set, is_train=False), dim=1)
