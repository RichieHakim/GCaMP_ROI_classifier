import torch
from torch.nn import Module
import time

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
    def __init__(self, dim=0, n_channels=3):
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

class WarpPoints(Module):
    """
    Warps the input tensor at the given points by the given deltas.
    RH 2021 / JZ 2021
    """
    
    def __init__(self,  r=[0, 2],
                        cx=[-0.5, 0.5],
                        cy=[-0.5, 0.5], 
                        dx=[-0.3, 0.3], 
                        dy=[-0.3, 0.3], 
                        n_warps=1,
                        prob=0.5,
                        img_size_in=[36, 36],
                        img_size_out=[36, 36]):
        """
        Initializes the class.

        Args:
            r (list):
                The range of the radius.
            cx (list):
                The range of the center x.
            cy (list):  
                The range of the center y.
            dx (list):
                The range of the delta x.
            dy (list):
                The range of the delta y.
            n_warps (int):
                The number of warps to apply.
            prob (float):
                The probability of adding noise at all.
            img_size_in (list):
                The size of the input image.
            img_size_out (list):
                The size of the output image.
        """
        tik = time.time()
        
        super().__init__()

        self.r = r
        self.cx = cx
        self.cy = cy
        self.dx = dx
        self.dy = dy
        self.n_warps = n_warps

        self.prob = prob

        self.img_size_in = img_size_in
        self.img_size_out = img_size_out

        self.r_range = r[1] - r[0]
        self.cx_range = cx[1] - cx[0]
        self.cy_range = cy[1] - cy[0]
        self.dx_range = dx[1] - dx[0]
        self.dy_range = dy[1] - dy[0]

        self.meshgrid_in =  torch.tile(torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img_size_in[0]),  torch.linspace(-1, 1, self.img_size_in[1])), dim=0)[...,None], (1,1,1, n_warps))
        self.meshgrid_out = torch.tile(torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img_size_out[0]), torch.linspace(-1, 1, self.img_size_out[1])), dim=0)[...,None], (1,1,1, n_warps))
        
        tok = time.time()
        print('Warp Initialization took:', tok-tik, 's')

    def gaus2D(self, x, y, sigma):
        return torch.exp(-((torch.square(self.meshgrid_out[0] - x[None,None,:]) + torch.square(self.meshgrid_out[1] - y[None,None,:]))/(2*torch.square(sigma[None,None,:]))))        

    def forward(self, tensor):
        tensor = tensor[None, ...]
        
        if torch.rand(1) <= self.prob:
            rands = torch.rand(5, self.n_warps)
            cx = rands[0,:] * (self.cx_range) + self.cx[0]
            cy = rands[1,:] * (self.cy_range) + self.cy[0]
            dx = rands[2,:] * (self.dx_range) + self.dx[0]
            dy = rands[3,:] * (self.dy_range) + self.dy[0]
            r =  rands[4,:] * (self.r_range)  + self.r[0]
            im_gaus = self.gaus2D(x=cx, y=cy, sigma=r) # shape: (img_size_x, img_size_y, n_warps)
            im_disp = im_gaus[None,...] * torch.stack([dx, dy], dim=0).reshape(2, 1, 1, self.n_warps) # shape: (2(dx,dy), img_size_x, img_size_y, n_warps)
            im_disp_composite = torch.sum(im_disp, dim=3, keepdim=True) # shape: (2(dx,dy), img_size_x, img_size_y)
            im_newPos = self.meshgrid_out[...,0:1] + im_disp_composite
        else:
            im_newPos = self.meshgrid_out[...,0:1]
        
        im_newPos = torch.permute(im_newPos, [3,2,1,0]) # Requires 1/2 transpose because otherwise results are transposed from torchvision Resize
        ret = torch.nn.functional.grid_sample( tensor, 
                                                im_newPos, 
                                                mode='bilinear',
                                                # mode='bicubic', 
                                                padding_mode='zeros', 
                                                align_corners=True)
        ret = ret[0]
        return ret
        
    def __repr__(self):
        return f"WarpPoints(r={self.r}, cx={self.cx}, cy={self.cy}, dx={self.dx}, dy={self.dy}, n_warps={self.n_warps}, prob={self.prob}, img_size_in={self.img_size_in}, img_size_out={self.img_size_out})"