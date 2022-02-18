import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_stat_and_make_spatialFootprints(path_list, stat_fileName='stat.npy', preCrop_length=241, crop_length=36, plot_pref=True):
    """
    RH 2021
    demo:
    from pathlib import Path

    path_list = list([
        Path(r'/media/rich/bigSSD/downloads_tmp/labels_ROI_Classifier/mouse 6_28 _ day 20200903'),
        Path(r'/media/rich/bigSSD/downloads_tmp/labels_ROI_Classifier/mouse6_28 _ day20200815'),
        Path(r'/media/rich/bigSSD/downloads_tmp/labels_ROI_Classifier/mouseUnknownAi148__20210325'),
        Path(r'/media/rich/bigSSD/downloads_tmp/labels_ROI_Classifier/mouse2_6__20210409'),
    ])

    images = load_stat_and_make_spatialFootprints(path_list, stat_fileName='stat.npy', preCrop_length=241, crop_length=36, plot_pref=True)
    """
    preCrop_center = int((preCrop_length-1)/2)
    spatial_footprints_centered_crop = list(np.zeros(len(path_list)))
    for ii in range(len(path_list)):
        stat = np.load(path_list[ii] / Path(stat_fileName), allow_pickle=True)
        print('stat file loaded')
        print('')

        num_ROI = stat.shape[0]
        print(f'Number of ROIs: {num_ROI}')

        spatial_footprints_centered = np.zeros((num_ROI, preCrop_length,preCrop_length))
        for i in range(num_ROI):
            spatial_footprints_centered[i , stat[i]['ypix'] - np.int16(stat[i]['med'][0]) + preCrop_center, stat[i]['xpix'] - np.int16(stat[i]['med'][1]) + preCrop_center] = stat[i]['lam'] # this is formatted for coding ease (dim1: y pix) (dim2: x pix) (dim3: ROI#)
    #     spatial_footprints_centered_crop = spatial_footprints_centered[:, 102:138 , 102:138]
        spatial_footprints_centered_crop[ii] = spatial_footprints_centered[:, 
                                                                       int(preCrop_center-(crop_length/2)):int(preCrop_center+(crop_length/2)),
                                                                       int(preCrop_center-(crop_length/2)):int(preCrop_center+(crop_length/2))]
    if plot_pref:
        fig, axs = plt.subplots(len(path_list), figsize=(5,3*len(path_list)))
        for ii in range(len(axs)):
            axs[ii].imshow(np.max(spatial_footprints_centered_crop[ii] , axis=0) ** 0.2);
        fig.suptitle(f'spatial_footprints_centered_crop MIP^0.2');

    return spatial_footprints_centered_crop


def load_labels(path_list, images, label_fileName='label', fileName_is_prefix=True, plot_pref=True):
    labels = list(np.zeros(len(path_list)))
    for ii in range(len(path_list)):
        if fileName_is_prefix:
            labels[ii] = np.load(list((path_list[ii].glob(f'{label_fileName}*')))[-1], allow_pickle=True)
        else:
            labels[ii] = np.load(path_list[ii]/label_fileName, allow_pickle=True)

        print('labels file loaded')
        print('')

        num_ROI = labels[ii].shape[0]
        print(f'Number of ROIs: {num_ROI}')

        plt.figure()
        plt.hist(labels[ii],20);

        #check in number of labels matches number of images
        if images[ii].shape[0] == len(labels[ii]):
            print('number of labels and images match')
        else:
            print('WARNING: number of labels and images DO NOT match')

    return labels