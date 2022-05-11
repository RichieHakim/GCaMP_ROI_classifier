# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))

## Import general libraries
from pathlib import Path
import os
import sys
import copy

import numpy as np
import itertools

### Import personal libraries
dir_github = '/media/rich/Home_Linux_partition/github_repos'
import sys
sys.path.append(dir_github)
# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import container_helpers, server


## set paths
dir_save = '/media/rich/bigSSD/'

path_script = '/media/rich/Home_Linux_partition/github_repos/GCaMP_ROI_classifier/scripts/20220508_simCLR/train_ROInet_simCLR_20220508.py'



## define params
params_template = {
    'paths': {
        'dir_github':'/media/rich/Home_Linux_partition/github_repos',
        'fileName_save_model':'EfficientNet_b0_7unfrozen_simCLR',
        'path_data_training':'/media/rich/bigSSD/analysis_data/ROIs_for_training/sf_sparse_36x36_20220503.npz',
    },
    
    'prefs': {
        'saveModelIteratively':True,
        'saveLogs':True,
    },
    
    'useGPU_training': True,
    'useGPU_dataloader': False,
    'dataloader_kwargs':{
        'batch_size': 1024,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 36,
        'persistent_workers': True,
        'prefetch_factor': 2,
    },

    'torchvision_model': 'convnext_tiny',

    'pre_head_fc_sizes': [128, 128],
    'post_head_fc_sizes': [128],
    'block_to_unfreeze': '1.2',
    'n_block_toInclude': 4,
    
    'lr': 1*10**-4,
    'weight_decay': 0.0000,
    'gamma': 1-0.0000,
    'n_epochs': 9999999,
    'temperature': 0.5,
    'l2_alpha': 0.0000,
    
    'augmentation': {
        'Scale_image_sum': {'sum_val':1, 'epsilon':1e-9, 'min_sub':True},
        'AddPoissonNoise': {'scaler_bounds':(10**(4), 10**(5)), 'prob':0.5, 'base':1000, 'scaling':'log'},
        'Horizontal_stripe_scale': {'alpha_min_max':(0.5, 1), 'im_size':(36,36), 'prob':0.5},
        'Horizontal_stripe_shift': {'alpha_min_max':(1  , 3), 'im_size':(36,36), 'prob':0.5},
        'RandomHorizontalFlip': {'p':0.5},
        'RandomAffine': {
            'degrees':(-180,180),
            'translate':(0.1, 0.1), #0, .3, .45 (DEFAULT)
            'scale':(0.6, 1.2), # no scale (1,1), (0.4, 1.5)
            'shear':(-15, 15, -15, 15),
#             'interpolation':torchvision.transforms.InterpolationMode.BILINEAR, 
            'interpolation':'bilinear', 
            'fill':0, 
            'fillcolor':None, 
            'resample':None,
        },
        'AddGaussianNoise': {'mean':0, 'std':0.0010, 'prob':0.5},
        'ScaleDynamicRange': {'scaler_bounds':(0,1), 'epsilon':1e-9},
        'WarpPoints': {
            'r':[0.3, 0.6],
            'cx':[-0.3, 0.3],
            'cy':[-0.3, 0.3], 
            'dx':[-0.24, 0.24], 
            'dy':[-0.24, 0.24], 
            'n_warps':2,
            'prob':0.5,
            'img_size_in':[36, 36],
            'img_size_out':[72,72],
        },
        'TileChannels': {'dim':0, 'n_channels':3},
    },
}



## make params dicts with grid swept values
params = copy.deepcopy(params_template)
params = [container_helpers.deep_update_dict(params, ['dataloader_kwargs', 'prefetch_factor'], val) for val in [4,5,6]]
params = container_helpers.flatten_list([[container_helpers.deep_update_dict(p, ['lr'], val) for val in [0.00001, 0.0001, 0.001]] for p in params])

params_unchanging, params_changing = container_helpers.find_differences_across_dictionaries(params)



## copy script .py file to dir_save
import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));



## save parameters to file
parameters_batch = {
    'params': params,
    'params_unchanging': params_unchanging,
    'params_changing': params_changing
}
import json
with open(str(Path(dir_save) / 'parameters_batch.json'), 'w') as f:
    json.dump(parameters_batch, f)

# with open(str(Path(dir_save) / 'parameters_batch.json')) as f:
#     test = json.load(f)



## define slurm SBATCH parameters
sbatch_config_default = \
"""#!/usr/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=/home/rh183/script_logs/python_01_%j.log
#SBATCH --partition=priority
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=1GB
#SBATCH --time=0-00:00:10

python "$@"
"""

## run batch_run function
paths_scripts = [path_script]
params_list = params
sbatch_config_list = [sbatch_config_default]
max_n_jobs=3
dir_save='/n/data1/hms/neurobio/sabatini/rich/analysis/test_dispatcher_ROInet'
name_save='jobNum_'

server.batch_run(paths_scripts=paths_scripts,
                    params_list=params_list,
                    sbatch_config_list=sbatch_config_list,
                    max_n_jobs=2,
                    dir_save=dir_save,
                    name_save='jobNum_',
                    verbose=True,
                    )
