import torch
from torch.nn import Module

class AddGaussianNoise(Module):
    """
    Adds Gaussian noise to the input tensor.
    RH 2021
    """
    def __init__(self, mean=0., std=1., level_bounds=(0., 1.)):
        """
        Initializes the class.

        Args:
            mean (float): 
                The mean of the Gaussian noise.
            std (float):
                The standard deviation of the Gaussian noise.
        """
        super().__init__()

        self.std = std
        self.mean = mean
        
        self.level_bounds = level_bounds
        self.level_range = level_bounds[1] - level_bounds[0]
    def forward(self, tensor):
        level = torch.rand(1) * self.level_range + self.level_bounds[0]
        return (1-level)*tensor + level*(tensor + torch.randn(tensor.shape) * self.std + self.mean)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class AddPoissonNoise(Module):
    """
    Adds Poisson noise to the input tensor.
    RH 2021
    """
    def __init__(self, level_bounds=(0.,0.3)):
        """
        Initializes the class.

        Args:
            lam (float): 
                The lambda parameter of the Poisson noise.
        """
        super().__init__()

        self.level_bounds = level_bounds
        self.level_range = level_bounds[1] - level_bounds[0]
    def forward(self, tensor):
        level = torch.rand(1) * self.level_range + self.level_bounds[0]
        return (1-level)*tensor + level*torch.poisson(tensor)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(lam={self.lam})'