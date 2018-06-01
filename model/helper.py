from model.mammogram_densenet import MammogramDenseNet

def get_tiny_densenet():
    return MammogramDenseNet(block_config=(6,))

def get_small_densenet():
    return MammogramDenseNet(block_config=(10, 6))

def get_medium_densenet():
    return MammogramDenseNet(block_config=(8, 12, 8))

def get_large_densenet():
    return MammogramDenseNet(block_config = (6,12,18,12)) # Default block config