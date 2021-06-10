from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def cv_split_datasets(images, labels, test_fraction=0.15, rand_seed=None):
    n_datasets = len(images)

    train_x, val_x, train_y, val_y, train_idx, val_idx = list(np.zeros(n_datasets)), list(np.zeros(n_datasets)), list(np.zeros(n_datasets)), list(np.zeros(n_datasets)), list(np.zeros(n_datasets)), list(np.zeros(n_datasets))
    for ii in range(n_datasets):
        train_x[ii], val_x[ii], train_y[ii], val_y[ii], train_idx[ii], val_idx[ii] = train_test_split(images[ii],
                                                                                                      labels[ii],
                                                                                                      np.arange(len(labels[ii])),
                                                                                                      test_size=test_fraction,
                                                                                                      random_state=rand_seed)
    train_x_all = np.concatenate(train_x)
    val_x_all = np.concatenate(val_x)
    train_y_all = np.concatenate(train_y)
    val_y_all = np.concatenate(val_y)

    return train_x, val_x, train_y, val_y, train_idx, val_idx,    train_x_all, val_x_all, train_y_all, val_y_all
