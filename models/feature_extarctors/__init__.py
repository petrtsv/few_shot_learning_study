from models.feature_extarctors.convnet import ConvNet64, ConvNet256

FEATURE_EXTRACTORS = {
    'convnet64': ConvNet64(),
    'convnet256': ConvNet256(),

    'convnet64pooling': ConvNet64(pooling=True),
    'convnet256pooling': ConvNet256(pooling=True),
}
