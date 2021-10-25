import torch
from torch.nn import Module

class AddGaussianNoise(Module):
    """
    Adds Gaussian noise to the input tensor.
    RH 2021
    """
    def __init__(self, mean=0., std=1., level_bounds=(0., 1.), prob=1):
        """
        Initializes the class.

        Args:
            mean (float): 
                The mean of the Gaussian noise.
            std (float):
                The standard deviation of the Gaussian 
                 noise.
            level_bounds (tuple):
                The lower and upper bound of how much
                 noise to add.
            prob (float):
                The probability of adding noise at all.
        """
        super().__init__()

        self.std = std
        self.mean = mean

        self.prob = prob
        
        self.level_bounds = level_bounds
        self.level_range = level_bounds[1] - level_bounds[0]
    def forward(self, tensor):
        if torch.rand(1) <= self.prob:
            level = torch.rand(1) * self.level_range + self.level_bounds[0]
            return (1-level)*tensor + level*(tensor + torch.randn(tensor.shape) * self.std + self.mean)
        else:
            return tensor
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std}, level_bounds={self.level_bounds}, prob={self.prob})"

class AddPoissonNoise(Module):
    """
    Adds Poisson noise to the input tensor.
    RH 2021
    """
    def __init__(self, level_bounds=(0.,0.3), prob=1):
        """
        Initializes the class.

        Args:
            lam (float): 
                The lambda parameter of the Poisson noise.
            level_bounds (tuple):
                The lower and upper bound of how much 
                 noise to add.
            prob (float):
                The probability of adding noise at all.
        """
        super().__init__()

        self.level_bounds = level_bounds
        self.level_range = level_bounds[1] - level_bounds[0]

        self.prob = prob
    def forward(self, tensor):
        if torch.rand(1) <= self.prob:
            level = torch.rand(1) * self.level_range + self.level_bounds[0]
            return (1-level)*tensor + level*torch.poisson(tensor)
        else:
            return tensor
    
    def __repr__(self):
        return f"AddPoissonNoise(level_bounds={self.level_bounds}, prob={self.prob})"