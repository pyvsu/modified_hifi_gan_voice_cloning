import torch
import torch.nn as nn

def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01):
    """
    Initializes convolutional weights with values from the normal distribution (mean = 0, std = 0.01)

    If this is not overridden, conv layers use Kaiming uniform initialization by default.
    
    HiFi-GAN overrides this because
    - GANs are very sensitive to initialization
    - The authors found that small random normal weights stabilize early training and reduce artifacts
    """

    classname = module.__class__.__name__

    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)
