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
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
            prob (float):
                The probability of adding noise at all.
            base (float):
                The base of the logarithm used if scaling
                 is set to 'log'. Larger base means more
                 noise (higher probability of scaler being
                 close to scaler_bounds[0]).
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

class ScaleDynamicRange(Module):
    """
    Scales the dynamic range of the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0,1)):
        """
        Initializes the class.

        Args:
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
    
    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / tensor_minSub.max())
    def __repr__(self):
        return f"ScaleDynamicRange(scaler_bounds={self.scaler_bounds})"

class TileChannels(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    RH 2021
    """
    def __init__(self, dim=-3, n_channels=3):
        """
        Initializes the class.

        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels

    def forward(self, tensor):
        dims = [1]*len(tensor.shape)
        dims[self.dim] = self.n_channels
        return torch.tile(tensor, dims)
    def __repr__(self):
        return f"TileChannels(dim={self.dim})"

class Normalize(Module):
    """
    Normalizes the input tensor by setting the 
     mean and standard deviation of each channel.
    RH 2021
    """
    def __init__(self, means=0, stds=1):
        """
        Initializes the class.

        Args:
            mean (float):
                Mean to set.
            std (float):
                Standard deviation to set.
        """
        super().__init__()
        self.means = torch.as_tensor(means)[:,None,None]
        self.stds = torch.as_tensor(stds)[:,None,None]
    def forward(self, tensor):
        tensor_means = tensor.mean(dim=(1,2), keepdim=True)
        tensor_stds = tensor.std(dim=(1,2), keepdim=True)
        tensor_z = (tensor - tensor_means) / tensor_stds
        return (tensor_z * self.stds) + self.means
