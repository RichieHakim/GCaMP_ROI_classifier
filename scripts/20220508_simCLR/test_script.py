# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))

# # !source activate jupyter_launcher
# !pip3 install numba
# !pip3 install matplotlib
# !pip3 install scipy
# !pip3 install torch
# !pip3 install torchvision
# !pip3 install sklearn
# !pip3 install pycuda
# !pip3 install tqdm
# !pip3 install seaborn
# !pip3 install h5py
# !pip3 install hdfdict
# !pip3 install ipywidgets
# !pip3 install numpy==1.20

import sys
print(sys.version_info)

import os
print(os.environ['CONDA_DEFAULT_ENV'])

import copy
import pathlib
from pathlib import Path
import time
import gc

from tqdm import tqdm, trange

import numpy as np
import scipy

import torch
import torchvision
import torchvision.transforms as transforms



## Parse arguments

import sys
path_script, path_params, dir_save = sys.argv
dir_save = Path(dir_save)
                
import json
with open(path_params, 'r') as f:
    params = json.load(f)

import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));




### Import personal libraries

import sys

sys.path.append(params['paths']['dir_github'])

# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import torch_helpers, math_functions, classification, h5_handling, plotting_helpers, indexing, misc, decomposition, path_helpers

def write_to_log(path_log, text, mode='a', start_on_new_line=True):
    with open(path_log, mode=mode) as log:
        if start_on_new_line==True:
            log.write('\n')
        log.write(text)





### Prepare paths

path_saveModel = str((dir_save / params['paths']['fileName_save_model']).with_suffix('.pth'))
path_saveLog = str(dir_save / 'log.txt')
path_saveLoss = (dir_save / 'loss.npy')

device_train = torch_helpers.set_device(use_GPU=params['useGPU_training'], verbose=False)

print(device_train)

test_ims = torch.rand(2000,36,36)

print(test_ims.to(device_train).device)
print(torch.rand(2,3, device=device_train) * torch.rand(2,3, device=device_train))
