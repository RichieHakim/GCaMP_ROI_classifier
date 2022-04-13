
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

###############################################################################
############################### MAKE DATALOADER ###############################
###############################################################################


class dataset_simCLR(Dataset):
    """    
    demo:
    
    transforms = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    
    torchvision.transforms.GaussianBlur(5,
                                        sigma=(0.01, 1.)),
    
    torchvision.transforms.RandomPerspective(distortion_scale=0.6, 
                                             p=1, 
                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                                             fill=0),
    torchvision.transforms.RandomAffine(
                                        degrees=(-180,180),
                                        translate=(0.4, 0.4),
                                        scale=(0.7, 1.7), 
                                        shear=(-20, 20, -20, 20), 
                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                                        fill=0, 
                                        fillcolor=None, 
                                        resample=None),
    )
    scripted_transforms = torch.jit.script(transforms)

    dataset = util.dataset_simCLR(  torch.tensor(images), 
                                labels, 
                                n_transforms=2, 
                                transform=scripted_transforms,
                                DEVICE='cpu',
                                dtype_X=torch.float32,
                                dtype_y=torch.int64 )
    
    dataloader = torch.utils.data.DataLoader(   dataset,
                                            batch_size=64,
        #                                     sampler=sampler,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=False,
                                            num_workers=0,
                                            )
    """
    def __init__(   self, 
                    X, 
                    y, 
                    n_transforms=2,
                    class_weights=None,
                    transform=None,
                    DEVICE='cpu',
                    dtype_X=torch.float32,
                    dtype_y=torch.int64,
                    temp_uncertainty=1,
                    expand_dim=True
                    ):

        """
        Make a dataset from a list / numpy array / torch tensor
        of images and labels.
        RH 2021 / JZ 2021

        Args:
            X (torch.Tensor / np.array / list of float32):
                Images.
                Shape: (n_samples, height, width)
                Currently expects no channel dimension. If/when
                 it exists, then shape should be
                (n_samples, n_channels, height, width)
            y (torch.Tensor / np.array / list of ints):
                Labels.
                Shape: (n_samples)
            n_transforms (int):
                Number of transformations to apply to each image.
                Should be >= 1.
            transform (callable, optional):
                Optional transform to be applied on a sample.
                See torchvision.transforms for more information.
                Can use torch.nn.Sequential( a bunch of transforms )
                 or other methods from torchvision.transforms. Try
                 to use torch.jit.script(transform) if possible.
                If not None:
                 Transform(s) are applied to each image and the 
                 output shape of X_sample_transformed for 
                 __getitem__ will be
                 (n_samples, n_transforms, n_channels, height, width)
                If None:
                 No transform is applied and output shape
                 of X_sample_trasformed for __getitem__ will be 
                 (n_samples, n_channels, height, width)
                 (which is missing the n_transforms dimension).
            DEVICE (str):
                Device on which the data will be stored and
                 transformed. Best to leave this as 'cpu' and do
                 .to(DEVICE) on the data for the training loop.
            dtype_X (torch.dtype):
                Data type of X.
            dtype_y (torch.dtype):
                Data type of y.
        
        Returns:
            torch.utils.data.Dataset:
                torch.utils.data.Dataset object.
        """

        self.expand_dim = expand_dim
        
        self.X = torch.as_tensor(X, dtype=dtype_X, device=DEVICE) # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.X = self.X[:,None,...] if expand_dim else self.X
        self.y = torch.as_tensor(y, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.
        
        self.idx = torch.arange(self.X.shape[0], device=DEVICE)
        self.n_samples = self.X.shape[0]

        self.transform = transform
        self.n_transforms = n_transforms

        self.temp_uncertainty = temp_uncertainty

        self.headmodel = None

        self.net_model = None
        self.classification_model = None
        
        
        # self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=DEVICE)

        # self.classModelParams_coef_ = mp.Array(np.ctypeslib.as_array(mp.Array(ctypes.c_float, feature)))

        if X.shape[0] != y.shape[0]:
            raise ValueError('RH Error: X and y must have same first dimension shape')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves and transforms a sample.
        RH 2021 / JZ 2021

        Args:
            idx (int):
                Index / indices of the sample to retrieve.
            
        Returns:
            X_sample_transformed (torch.Tensor):
                Transformed sample(s).
                Shape: 
                    If transform is None:
                        X_sample_transformed[batch_size, n_channels, height, width]
                    If transform is not None:
                        X_sample_transformed[n_transforms][batch_size, n_channels, height, width]
            y_sample (int):
                Label(s) of the sample(s).
            idx_sample (int):
                Index of the sample(s).
        """

        y_sample = self.y[idx]
        idx_sample = self.idx[idx]
        
        if self.classification_model is not None:
            # features = self.net_model(tile_channels(self.X[idx][:,None,...], dim=1))
            # proba = self.classification_model.predict_proba(features.cpu().detach())[0]
            proba = self.classification_model.predict_proba(tile_channels(self.X[idx_sample][:,None,...], dim=-3))[0]
            
            # sample_weight = loss_uncertainty(torch.as_tensor(proba, dtype=torch.float32), temperature=self.temp_uncertainty)
            sample_weight = 1
        else:
            sample_weight = 1

        X_sample_transformed = []
        if self.transform is not None:
            for ii in range(self.n_transforms):

                # X_sample_transformed.append(tile_channels(self.transform(self.X[idx_sample]), dim=0))
                X_transformed = self.transform(self.X[idx_sample])
                X_sample_transformed.append(X_transformed)
        else:
            X_sample_transformed = tile_channels(self.X[idx_sample], dim=-3)
        
        return X_sample_transformed, y_sample, idx_sample, sample_weight



def loss_uncertainty(proba, temperature=1, class_value=None):
    # if class_value is None:
    #     class_value = torch.ones(proba.shape[1], dtype=proba.dtype)
    # return (proba @ class_value)/(torch.linalg.norm(proba)**temperature)
    return 1/(torch.linalg.norm(proba)**temperature)


def tile_channels(X_in, dim=-3):
    """
    Expand dimension dim in X_in and tile to be 3 channels.

    JZ 2021 / RH 2021

    Args:
        X_in (torch.Tensor or np.ndarray):
            Input image. 
            Shape: [n_channels==1, height, width]

    Returns:
        X_out (torch.Tensor or np.ndarray):
            Output image.
            Shape: [n_channels==3, height, width]
    """
    dims = [1]*len(X_in.shape)
    dims[dim] = 3
    return torch.tile(X_in, dims)




class dataset_supervised(Dataset):
    """
    """
    def __init__(   self, 
                    X, 
                    y, 
                    # n_transforms=2,
                    class_weights=None,
                    transform=None,
                    DEVICE='cpu',
                    dtype_X=torch.float32,
                    dtype_y=torch.int64):

        """
        Make a dataset from a list / numpy array / torch tensor
        of images and labels.
        RH 2021 / JZ 2021

        Args:
            X (torch.Tensor / np.array / list of float32):
                Images.
                Shape: (n_samples, height, width)
                Currently expects no channel dimension. If/when
                 it exists, then shape should be
                (n_samples, n_channels, height, width)
            y (torch.Tensor / np.array / list of ints):
                Labels.
                Shape: (n_samples)
            n_transforms (int):
                Number of transformations to apply to each image.
                Should be >= 1.
            transform (callable, optional):
                Optional transform to be applied on a sample.
                See torchvision.transforms for more information.
                Can use torch.nn.Sequential( a bunch of transforms )
                 or other methods from torchvision.transforms. Try
                 to use torch.jit.script(transform) if possible.
                If not None:
                 Transform(s) are applied to each image and the 
                 output shape of X_sample_transformed for 
                 __getitem__ will be
                 (n_samples, n_transforms, n_channels, height, width)
                If None:
                 No transform is applied and output shape
                 of X_sample_trasformed for __getitem__ will be 
                 (n_samples, n_channels, height, width)
                 (which is missing the n_transforms dimension).
            DEVICE (str):
                Device on which the data will be stored and
                 transformed. Best to leave this as 'cpu' and do
                 .to(DEVICE) on the data for the training loop.
            dtype_X (torch.dtype):
                Data type of X.
            dtype_y (torch.dtype):
                Data type of y.
        
        Returns:
            torch.utils.data.Dataset:
                torch.utils.data.Dataset object.
        """

        self.X = torch.as_tensor(X, dtype=dtype_X, device=DEVICE)[:,None,...] # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.y = torch.as_tensor(y, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.
        
        self.idx = torch.arange(self.X.shape[0], device=DEVICE)
        self.n_samples = self.X.shape[0]

        self.transform = transform
        # self.n_transforms = n_transforms

        # self.headmodel = None

        # self.net_model = None
        # self.classification_model = None
        # self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=DEVICE)

        # self.classModelParams_coef_ = mp.Array(np.ctypeslib.as_array(mp.Array(ctypes.c_float, feature)))

        if X.shape[0] != y.shape[0]:
            raise ValueError('RH Error: X and y must have same first dimension shape')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves and transforms a sample.
        RH 2021 / JZ 2021

        Args:
            idx (int):
                Index / indices of the sample to retrieve.
            
        Returns:
            X_sample_transformed (torch.Tensor):
                Transformed sample(s).
                Shape: 
                    If transform is None:
                        X_sample_transformed[batch_size, n_channels, height, width]
                    If transform is not None:
                        X_sample_transformed[n_transforms][batch_size, n_channels, height, width]
            y_sample (int):
                Label(s) of the sample(s).
            idx_sample (int):
                Index of the sample(s).
        """

        y_sample = self.y[idx]
        idx_sample = self.idx[idx]

        X_sample_transformed = []
        if self.transform is not None:
            X_sample_transformed = tile_channels(self.transform(self.X[idx_sample]), dim=0)
        else:
            X_sample_transformed = tile_channels(self.X[idx_sample], dim=0)
        
        return X_sample_transformed, y_sample, idx_sample

###############################################################################
############################## Model helpers ##################################
###############################################################################
