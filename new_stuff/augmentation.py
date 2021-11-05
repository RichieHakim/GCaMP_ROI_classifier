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
            level = torch.rand(1, device=tensor.device) * self.level_range + self.level_bounds[0]
            return (1-level)*tensor + level*(tensor + torch.randn(tensor.shape, device=tensor.device) * self.std + self.mean)
        else:
            return tensor
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std}, level_bounds={self.level_bounds}, prob={self.prob})"

class AddPoissonNoise(Module):
    """
    Adds Poisson noise to the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0.1,1), prob=1, base=10, scaling='log'):
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
            scaler (float):
                How much to scale the input by when passing
                 torch.poisson; then undo the rescale in 
                 the output.
                Smaller values result in noisier outputs.\
            scaling (str):
                'linear' or 'log'
        """
        super().__init__()

        self.prob = prob
        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        self.base = base
        self.scaling = scaling

    def forward(self, tensor):
        if torch.rand(1) <= self.prob:
            if self.scaling == 'linear':
                scaler = torch.rand(1, device=tensor.device) * self.range + self.bounds[0]
                return torch.poisson(tensor * scaler) / scaler
            else:
                scaler = (((self.base**torch.rand(1, device=tensor.device) - 1)/(self.base-1)) * self.range) + self.bounds[0]
                return torch.poisson(tensor * scaler) / scaler
        else:
            return tensor
    
    def __repr__(self):
        return f"AddPoissonNoise(level_bounds={self.level_bounds}, prob={self.prob})"