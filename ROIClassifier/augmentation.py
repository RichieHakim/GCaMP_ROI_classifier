import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

import skimage.transform


def gaussian_kernel_2D(center = (5, 5), image_size = (11, 11), sig = 1):
    """
    Generate a 2D or 1D gaussian kernel
    RH 2021

    Args:
        center (tuple):  the mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D gaussian
        image_size (tuple): The total image size (width, height). Make second value 0 to make 1D gaussian
        sig (scalar): The sigma value of the gaussian
    
    Return:
        kernel (np.ndarray): 2D or 1D array of the gaussian kernel
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel


def freq_filt_2D(input_image, window_image, plot_pref=True):
    im_fft = fftshift(fft2(input_image))

    im_fft_win = window_image * im_fft

    im_fft_win_ifft = ifft2(im_fft_win)

    if plot_pref:
        plt.figure()
        plt.imshow(input_image)

        plt.figure()
        plt.imshow(np.abs(im_fft))
        plt.colorbar()

        plt.figure()
        plt.imshow(np.log(np.abs(im_fft_win)))

        plt.figure()
        plt.imshow(abs(im_fft_win_ifft))


def gaussian(x, mu, sig , plot_pref=False):
    '''
    A gaussian function (normalized similarly to scipy's function)
    RH 2021
    
    Args:
        x (np.ndarray): 1-D array of the x-axis of the kernel
        mu (float): center position on x-axis
        sig (float): standard deviation (sigma) of gaussian
        plot_pref (boolean): True/False or 1/0. Whether you'd like the kernel plotted
    Returns:
        gaus (np.ndarray): gaussian function (normalized) of x
        params_gaus (dict): dictionary containing the input params
    '''

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x-mu)/sig, 2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')
    
    params_gaus = {
        "x": x,
        "mu": mu,
        "sig": sig,
    }

    return gaus , params_gaus


def make_distance_image(center_idx, vid_height, vid_width):
    """
    creates a matrix of cartesian coordinate distances from the center

    Args:
        center_idx (list): chosen center index
        vid_height (int): height of the video in pixels
        vid_width (int): width of the video in pixels

    Returns:
        distance_image (np.ndarray): array of distances to the center index

    demo:
        kernel_length = 7 # must be odd
        kernel_size = 0.3
        kernel_upscaleFactor = 100
        kernel_centerIdx = (kernel_length*kernel_upscaleFactor-1)/2 
        dist_im = make_distance_image([kernel_centerIdx , kernel_centerIdx], kernel_length*kernel_upscaleFactor, kernel_length*kernel_upscaleFactor)

    """

    x, y = np.meshgrid(range(vid_width), range(vid_height))  # note dim 1:X and dim 2:Y
    return np.sqrt((y - int(center_idx[1])) ** 2 + (x - int(center_idx[0])) ** 2)


def scale_between(x, lower=0, upper=1, axes=0, lower_percentile=None, upper_percentile=None, crop_pref=True):
    '''
    Scales the first (or more) dimension of an array to be between 
    lower and upper bounds.
    RH 2021

    Args:
        x (ndarray):
            Any dimensional array. First dimension is scaled
        lower (scalar):
            lower bound for scaling
        upper (scalar):
            upper bound for scaling
        axes (tuple):
            UNTESTED for values other than 0.
            It should work for tuples defining axes, so long
            as the axes are sequential starting from 0
            (eg. (0,1,2,3) ).
        lower_percentile (scalar):
            Ranges between 0-100. Defines what percentile of
            the data to set as the lower bound
        upper_percentile (scalar):
            Ranges between 0-100. Defines what percentile of
            the data to set as the upper bound
        crop_pref (bool):
            If true then data is cropped to be between lower
            and upper. Only meaningful if lower_percentile or
            upper_percentile is not None.

    Returns:
        x_out (ndarray):
            Scaled array
    '''

    if lower_percentile is not None:
        lowest_val = np.percentile(x, lower_percentile, axis=axes)
    else:
        lowest_val = np.min(x, axis=axes)
    if upper_percentile is not None:
        highest_val = np.percentile(x, upper_percentile, axis=axes)
    else:
        highest_val = np.max(x, axis=axes)

    x_out = ((x - lowest_val) * (upper - lower) / (highest_val - lowest_val) ) + lower

    if crop_pref:
        x_out[x_out < lower] = lower
        x_out[x_out > upper] = upper

    return x_out


# useful for making manual filters (don't push through conv layers. side load into the later FC layers)
def make_cosine_taurus(offset, width):
    l = (offset + width)*2 + 1
    c_idx = (l-1)/2
    cosine = np.cos(np.linspace((-np.pi) , (np.pi), width)) + 1
    cosine = np.concatenate((np.zeros(offset), cosine))
    dist_im = make_distance_image([c_idx , c_idx], l, l)
    taurus = cosine[np.searchsorted(np.arange(len(cosine)), dist_im, side='left')-1]
    return taurus


def make_mexicanHat_kernel(kernel_length=31, mexHat_width=2.5, plot_pref=True):
    if kernel_length%2 != 1:
        raise NameError('kernel_length should be odd')
    kernel_centerIdx = (kernel_length-1)/2 
    gaus , params_gaus = gaussian(np.arange(-kernel_centerIdx, kernel_centerIdx+1), 0, mexHat_width)
    mexHat = -np.diff(gaus,2)
    x_mexHat = params_gaus['x'][1:-1]

    if plot_pref:
        plt.figure()
        plt.plot(params_gaus['x'], gaus)
        plt.figure()
        plt.plot(x_mexHat, mexHat)

    return mexHat, x_mexHat


def make_mexHat_2D(kernel_length=7, kernel_size=0.3, kernel_upscaleFactor=101, plot_pref=True):

    if kernel_length%2 != 1:
        raise NameError('kernel_length should be odd')
    if kernel_upscaleFactor%2 != 1:
        raise NameError('kernel_upscaleFactor should be odd')    

    kernel_centerIdx = (kernel_length*kernel_upscaleFactor-1)/2 

    mexHat, x_mexHat = make_mexicanHat_kernel(kernel_length=kernel_length*kernel_upscaleFactor,
                                              mexHat_width=kernel_size*kernel_upscaleFactor,
                                              plot_pref=plot_pref)
    dist_im = make_distance_image([kernel_centerIdx , kernel_centerIdx], kernel_length*kernel_upscaleFactor, kernel_length*kernel_upscaleFactor)
    dist_im = (dist_im/(kernel_length*kernel_upscaleFactor))*2 * ((kernel_length-1)/2)

    if plot_pref:
        plt.figure();
        plt.imshow(dist_im);
        plt.colorbar();

    mexHat_2D_large = mexHat[np.searchsorted(x_mexHat/kernel_upscaleFactor, dist_im, side='left')-1]

    if plot_pref:
        plt.figure();
        plt.imshow(mexHat_2D_large);
        plt.colorbar();

    mexHat_2D = skimage.transform.resize(mexHat_2D_large, (kernel_length,kernel_length), 
                                         order=None, 
                                         anti_aliasing=False)
    mexHat_2D = scale_between(mexHat_2D, lower=0, upper=1, axes=(0,1))
    mexHat_2D = mexHat_2D - np.mean(mexHat_2D)

    if plot_pref:
        plt.figure();
        plt.imshow(mexHat_2D);
        plt.colorbar();