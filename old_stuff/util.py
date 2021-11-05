import pathlib
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import ctypes
import multiprocessing as mp


###############################################################################
############################## IMPORT STAT FILES ##############################
###############################################################################

def statFile_to_spatialFootprints(path_statFile=None, statFile=None, out_height_width=[36,36], max_footprint_width=241, plot_pref=True):
    """
    Converts a stat file to a list of spatial footprint images.
    RH 2021

    Args:
        path_statFile (pathlib.Path or str):
            Path to the stat file.
            Optional: if statFile is provided, this
             argument is ignored.
        statFile (dict):
            Suite2p stat file dictionary
            Optional: if path_statFile is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.
    
    Returns:
        sf_all (list):
            List of spatial footprints images
    """
    assert out_height_width[0]%2 == 0 and out_height_width[1]%2 == 0 , "RH: 'out_height_width' must be list of 2 EVEN integers"
    assert max_footprint_width%2 != 0 , "RH: 'max_footprint_width' must be odd"
    if statFile is None:
        stat = np.load(path_statFile, allow_pickle=True)
    else:
        stat = statFile
    n_roi = stat.shape[0]
    
    # sf_big: 'spatial footprints' prior to cropping. sf is after cropping
    sf_big_width = max_footprint_width # make odd number
    sf_big_mid = sf_big_width // 2

    sf_big = np.zeros((n_roi, sf_big_width, sf_big_width))
    for ii in range(n_roi):
        sf_big[ii , stat[ii]['ypix'] - np.int16(stat[ii]['med'][0]) + sf_big_mid, stat[ii]['xpix'] - np.int16(stat[ii]['med'][1]) + sf_big_mid] = stat[ii]['lam'] # (dim0: ROI#) (dim1: y pix) (dim2: x pix)

    sf = sf_big[:,  
                sf_big_mid - out_height_width[0]//2:sf_big_mid + out_height_width[0]//2,
                sf_big_mid - out_height_width[1]//2:sf_big_mid + out_height_width[1]//2]
    if plot_pref:
        plt.figure()
        plt.imshow(np.max(sf, axis=0)**0.2)
        plt.title('spatial footprints cropped MIP^0.2')
    
    return sf

def import_multiple_stat_files(paths_statFiles=None, dir_statFiles=None, fileNames_statFiles=None, out_height_width=[36,36], max_footprint_width=241, plot_pref=True):
    """
    Imports multiple stat files.
    RH 2021 
    
    Args:
        paths_statFiles (list):
            List of paths to stat files.
            Elements can be either str or pathlib.Path.
        dir_statFiles (str):
            Directory of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        fileNames_statFiles (list):
            List of file names of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.

    Returns:
        stat_all (list):
            List of stat files.
    """
    if paths_statFiles is None:
        paths_statFiles = [pathlib.Path(dir_statFiles) / fileName for fileName in fileNames_statFiles]

    sf_all_list = [statFile_to_spatialFootprints(path_statFile=path_statFile,
                                                 out_height_width=out_height_width,
                                                 max_footprint_width=max_footprint_width,
                                                 plot_pref=plot_pref)
                  for path_statFile in paths_statFiles]
    return sf_all_list

def convert_multiple_stat_files(statFiles_list=None, statFiles_dict=None, out_height_width=[36,36], max_footprint_width=241, print_pref=False, plot_pref=False):
    """
    Converts multiple stat files to spatial footprints.
    RH 2021

    Args:
        statFiles_list (list):
            List of stat files.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.
    """
    if statFiles_dict is None:
        sf_all_list = [statFile_to_spatialFootprints(statFile=statFile,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref)
                    for statFile in statFiles_list]
    else:
        sf_all_list = []
        for key, stat in statFiles_dict.items():
            if print_pref:
                print(key)
            sf_all_list.append(statFile_to_spatialFootprints(statFile=stat,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref))
    return sf_all_list
    

def import_multiple_label_files(paths_labelFiles=None, dir_labelFiles=None, fileNames_labelFiles=None, plot_pref=True):
    """
    Imports multiple label files.
    RH 2021

    Args:
        paths_labelFiles (list):
            List of paths to label files.
            Elements can be either str or pathlib.Path.
        dir_labelFiles (str):
            Directory of label files.
            Optional: if paths_labelFiles is provided, this
             argument is ignored.
        fileNames_labelFiles (list):
            List of file names of label files.
            Optional: if paths_labelFiles is provided, this
             argument is ignored.
        plot_pref (bool):
            If True, plots the label files.
    """
    if paths_labelFiles is None:
        paths_labelFiles = [pathlib.Path(dir_labelFiles) / fileName for fileName in fileNames_labelFiles]

    labels_all_list = [np.load(path_labelFile, allow_pickle=True) for path_labelFile in paths_labelFiles]

    if plot_pref:
        for ii, labels in enumerate(labels_all_list):
            plt.figure()
            plt.hist(labels, 20)
            plt.title('labels ' + str(ii))
    return labels_all_list


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
                    dtype_y=torch.int64):

        """
        Make a dataset from a list / numpy array / torch tensor
        of images and labels.
        RH 2021

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
        self.n_transforms = n_transforms

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
        RH 2021

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
            features = self.net_model(tile_channels(self.X[idx][:,None,...], dim=1))
            proba = self.classification_model.predict_proba(features.cpu().detach())[0]
            sample_weight = loss_uncertainty(torch.as_tensor(proba, dtype=torch.float32, device='cpu'), temperature=6)
        else:
            sample_weight = 1

        X_sample_transformed = []
        if self.transform is not None:
            for ii in range(self.n_transforms):

                X_sample_transformed.append(tile_channels(self.transform(self.X[idx_sample]), dim=0))
        else:
            X_sample_transformed = tile_channels(self.X[idx_sample], dim=0)
        
        return X_sample_transformed, y_sample, idx_sample, sample_weight



def loss_uncertainty(proba, temperature=1, class_value=None):
    # if class_value is None:
    #     class_value = torch.ones(proba.shape[1], dtype=proba.dtype)
    # return (proba @ class_value)/(torch.linalg.norm(proba)**temperature)
    return 1/(torch.linalg.norm(proba)**temperature)


def tile_channels(X_in, dim=0):
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