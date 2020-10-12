from torch import nn


class NoFlatteningBackbone(nn.Module):
    def __init__(self):
        super(NoFlatteningBackbone, self).__init__()

    def output_featmap_size(self):
        raise NotImplementedError()

    def output_features(self):
        raise NotImplementedError()
