from typing import Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.metrics import accuracy
from models.auxillary_tasks import RotationTask
from models.dfmn import DFMNLoss, ScaleModule
from models.feature_extarctors.base import NoFlatteningBackbone
from models.relevance_estimation import RelevanceEstimator
from models.set2set_adaptation import FEATTransformer
from utils import remove_dim


class FSLSolver(nn.Module):
    def __init__(self, backbone: NoFlatteningBackbone, k=1.0, aux_rotation_k=0.0, aux_location_k=0.0, dfmn_k=0.0,
                 dataset_classes=None, train_n_way=None, distance_type='cosine_scale', feat=False,
                 relevance_estimation=False):
        super(FSLSolver, self).__init__()
        self.feature_extractor = backbone

        self.is_feat = feat

        assert distance_type in ('cosine_scale', 'euclidean', 'sen2')
        self.distance_type = distance_type

        self.scale_module = ScaleModule(in_features=self.feature_extractor.output_features(),
                                        map_size=self.feature_extractor.output_featmap_size())

        self.dataset_classes = dataset_classes
        self.train_n_way = train_n_way

        self.k = k

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
                                      self.feature_extractor.output_featmap_size(), self.dataset_classes)

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

        self.feat_transformer = FEATTransformer(
            self.feature_extractor.output_features() * (self.feature_extractor.output_featmap_size() ** 2))

        self.relevance_estimation = relevance_estimation
        self.relevance_estimator = RelevanceEstimator(self.feature_extractor.output_features(),
                                                      self.feature_extractor.output_featmap_size())

    def extract_features(self, batch: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        MAX_BATCH_SIZE: int = 500
        batch_size: int = batch.size(0)
        sections: List[int] = []
        while batch_size > MAX_BATCH_SIZE:
            sections.append(MAX_BATCH_SIZE)
            batch_size -= MAX_BATCH_SIZE
        sections.append(batch_size)

        minibatches = batch.split(sections)
        xs = []
        for minibatch in minibatches:
            output = self.feature_extractor(minibatch)
            xs.append(output)
        features = torch.cat(xs)
        # features = self.feature_extractor(batch)

        if flatten:
            return features.view(features.size(0), -1)
        else:
            return features

    def extract_relevance_estimation(self, batch: torch.Tensor) -> torch.Tensor:
        MAX_BATCH_SIZE: int = 500
        batch_size: int = batch.size(0)
        sections: List[int] = []
        while batch_size > MAX_BATCH_SIZE:
            sections.append(MAX_BATCH_SIZE)
            batch_size -= MAX_BATCH_SIZE
        sections.append(batch_size)

        minibatches = batch.split(sections)
        xs = []
        for minibatch in minibatches:
            output = self.relevance_estimator(minibatch)
            xs.append(output)
        features = torch.cat(xs)
        # features = self.feature_extractor(batch)

        return features.view(features.size(0))

    def compute_prototypes(self, support_set: torch.Tensor, relevance: torch.Tensor):
        relevance_sum = relevance.sum(dim=1)
        relevance_expanded = relevance.unsqueeze(2).repeat(1, 1, support_set.size(2))

        scaled_support_set = support_set * relevance_expanded
        prototypes = scaled_support_set.sum(dim=1)
        prototypes = prototypes / (relevance_sum.unsqueeze(1).repeat(1, prototypes.size(1)))

        return prototypes

    @torch.jit.export
    def get_prototypes(self, support_set: torch.Tensor):
        n_classes = support_set.size(0)
        support_set_size = support_set.size(1)
        support_set_features = self.extract_features(remove_dim(support_set, 1)).view(n_classes,
                                                                                      support_set_size, -1)
        support_set_relevance = torch.ones(size=(n_classes, support_set_size), device=support_set_features.device)
        if self.relevance_estimation:
            support_set_relevance = self.extract_relevance_estimation(
                remove_dim(support_set_features, 1).view(-1, self.feature_extractor.output_features(),
                                                         self.feature_extractor.output_featmap_size(),
                                                         self.feature_extractor.output_featmap_size())).view(n_classes,
                                                                                                             support_set_size)

        class_prototypes = self.compute_prototypes(support_set_features, support_set_relevance)

        if self.is_feat:
            class_prototypes = self.feat_transformer(class_prototypes)

        return class_prototypes

    def distance(self, a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor = torch.tensor(0)):
        # a_scale = 1.0
        # b_scale = 1.0
        if self.distance_type == 'euclidean':
            return (a - b).pow(2).sum(dim=1).sqrt()
        elif self.distance_type == 'cosine_scale':
            a_scale = self.scale_module(
                a.view(-1, self.feature_extractor.output_features(), self.feature_extractor.output_featmap_size(),
                       self.feature_extractor.output_featmap_size()))
            b_scale = self.scale_module(
                b.view(-1, self.feature_extractor.output_features(), self.feature_extractor.output_featmap_size(),
                       self.feature_extractor.output_featmap_size()))
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            a = torch.div(a, a_scale)
            b = torch.div(b, b_scale)

            return (a - b).pow(2).sum(dim=1)
        elif self.distance_type == 'sen2':
            euclidean = (a - b).pow(2).sum(dim=1)
            norms = (a.norm(dim=1, p=2) - b.norm(dim=1, p=2)).pow(2)
            if not (torch.tensor(0).to(labels)).equal(labels):
                k = 1.0 * labels - (10 ** -7) * (1 - labels)
            else:
                k = torch.ones_like(norms)
            return (euclidean + k * norms).sqrt()
        else:
            raise NotImplementedError("Distance is not implemented yet")

    def scores(self, prototypes: torch.Tensor, query_set_features: torch.Tensor,
               is_train: bool = False, labels: torch.Tensor = torch.tensor(0)) -> torch.Tensor:
        n_classes = prototypes.size(0)
        query_set_size = query_set_features.size(0)

        class_prototypes = prototypes

        query_set_features_prepared = query_set_features.repeat_interleave(repeats=n_classes,
                                                                           dim=0)

        class_prototypes_expanded = class_prototypes.repeat(query_set_size, 1)

        if is_train and self.distance_type == 'sen':
            labels_expanded = labels.repeat_interleave(repeats=n_classes,
                                                       dim=0)
            prototypes_labels_expanded = torch.arange(start=0, end=n_classes, device=labels.device).repeat(
                query_set_size)

            binary_labels = (labels_expanded.eq(prototypes_labels_expanded)).float()
            distances = torch.stack(
                self.distance(
                    class_prototypes_expanded,
                    query_set_features_prepared,
                    labels=binary_labels
                ).split(n_classes)
            )
        else:
            distances = torch.stack(
                self.distance(
                    class_prototypes_expanded,
                    query_set_features_prepared,
                ).split(n_classes)
            )

        return -distances

    @torch.jit.ignore
    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor, labels: torch.Tensor,
                global_classes_mapping: dict) -> Tuple[torch.Tensor, dict, dict]:

        query_set_features = self.extract_features(query_set)

        prototypes = self.get_prototypes(support_set)

        scores = self.scores(prototypes=prototypes, query_set_features=query_set_features, is_train=True, labels=labels)

        loss = self.loss_fn(scores, labels) * self.k
        losses = {'fsl_loss': loss.item()}

        if self.aux_rotation_task is not None:
            combined_input = torch.cat((remove_dim(support_set, 1), query_set), 0)
            rotated, rotation_labels = self.aux_rotation_task.get_task(combined_input)
            rotated_features = self.extract_features(rotated, flatten=False)
            rot_loss = self.aux_rotation_task(rotated_features, rotation_labels)
            loss += rot_loss * self.aux_rotation_k
            losses['rotation_loss'] = rot_loss.item() * self.aux_rotation_k

        if self.dfmn_loss is not None:
            loss_dfmn = self.dfmn_loss(query_set_features.view(-1, self.feature_extractor.output_features(),
                                                               self.feature_extractor.output_featmap_size(),
                                                               self.feature_extractor.output_featmap_size()), labels,
                                       global_classes_mapping, self.train_n_way)
            loss += loss_dfmn * self.dfmn_k
            losses['dfmn_loss'] = loss_dfmn.item() * self.dfmn_k

        metrics = {'accuracy': accuracy(scores, labels)}

        return loss, losses, metrics

    @torch.jit.export
    def inference(self, prototypes: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        query_set_features = self.extract_features(query_set)
        return F.softmax(self.scores(prototypes, query_set_features, is_train=False), dim=1)
