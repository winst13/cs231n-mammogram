from model.mammogram_densenet import MammogramDenseNet

def get_tiny_densenet(**kwargs):
    return MammogramDenseNet(block_config=(6,), **kwargs)

def get_small_densenet(**kwargs):
    return MammogramDenseNet(block_config=(10, 6), **kwargs)

def get_medium_densenet(**kwargs):
    return MammogramDenseNet(block_config=(8, 12, 8), **kwargs)

def get_large_densenet(**kwargs):
    return MammogramDenseNet(block_config = (6,12,18,12), **kwargs) # Default block config