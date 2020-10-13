from data.cifar10 import CIFAR10Dataset, CIFAR100Dataset
from data.gtsrb import GTSRBDataset
from data.cub import CUBDataset
from data.mini_imagenet import MiniImageNetDataset
from data.google_landmarks import GoogleLandmarksDataset, GoogleLandmarksDataset2, \
    GoogleLandmarksDatasetSelfSupervision, GoogleLandmarksDatasetTest
from data.taco import TacoDataset

LABELED_DATASETS = {
    'cifar10': CIFAR10Dataset,
    'cifar100': CIFAR100Dataset,
    'gtsrb': GTSRBDataset,
    'cub': CUBDataset,
    'miniImageNet': MiniImageNetDataset,
    'taco': TacoDataset,
    'google-landmarks': GoogleLandmarksDataset,
    'google-landmarks-2': GoogleLandmarksDataset2,
    'google-landmarks-selfsupervision': GoogleLandmarksDatasetSelfSupervision,
    'google-landmarks-test': GoogleLandmarksDatasetTest,
}
