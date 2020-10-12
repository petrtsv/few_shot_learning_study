import os

# Default dataset directories
DEFAULT_DATASETS_DIR = r'C:\datasets'

DEFAULT_CUB_DIR = r'C:\datasets\CUB'
DEFAULT_GOOGLE_LANDMARKS_DIR = r'C:\datasets\google-landmarks'
DEFAULT_GTSRB_DIR = r'C:\datasets\gtsrb-german-traffic-sign'
DEFAULT_MINI_IMAGENET_DIR = r'C:\datasets\mini-imagenet'
DEFAULT_TACO_DIR = r'C:\datasets\taco'

# Experiments history

EXPERIMENTS_INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.csv')
EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
