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
# dir_github = '/media/rich/Home_Linux_partition/github_repos'
# dir_github = '/n/data1/hms/neurobio/sabatini/josh/github_repos'
dir_github = '/n/data1/hms/neurobio/sabatini/rich/github_repos'

import sys
sys.path.append(dir_github)
# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import container_helpers, server


## set paths
# dir_save = '/media/rich/bigSSD/'
# dir_save = '/n/data1/hms/neurobio/sabatini/josh/github_repos/GCaMP_ROI_classifier/scripts/outputs'
dir_save = '/n/data1/hms/neurobio/sabatini/rich/analysis/ROI_net_training/20220512_SimCLR_poolingMethod'
Path(dir_save).mkdir(parents=True, exist_ok=True)


# path_script = '/media/rich/Home_Linux_partition/github_repos/GCaMP_ROI_classifier/scripts/20220508_simCLR/train_ROInet_simCLR_20220508.py'
path_script = '/n/data1/hms/neurobio/sabatini/rich/github_repos/GCaMP_ROI_classifier/scripts/20220508_simCLR/train_ROInet_simCLR_20220508.py'


params_template = {
    'pref_log_all_steps': True,
    'paths': {
        # 'dir_github': '/media/rich/Home_Linux_partition/github_repos',
        'dir_github': dir_github,
        'fileName_save_model': 'ConvNext_tiny__1_0_unfrozen__simCLR',
        # 'path_data_training': '/media/rich/bigSSD/analysis_data/ROIs_for_training/sf_sparse_36x36_20220503.npz',
        'path_data_training': '/n/data1/hms/neurobio/sabatini/rich/data/ROI_network_data/sf_sparse_36x36_20220503.npz',
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
        # 'num_workers': 18,
        # 'persistent_workers': True,
        # 'prefetch_factor': 2,
        'num_workers': 4,
        'persistent_workers': True,
        'prefetch_factor': 1,
    },
    'inner_batch_size': 256,

    'torchvision_model': 'convnext_tiny',

    'head_pool_method': 'AdaptiveAvgPool2d',
    'head_pool_method_kwargs': {'output_size': 1},
    'pre_head_fc_sizes': [256, 128],
    'post_head_fc_sizes': [128],
    'block_to_unfreeze': '6.0',
    'n_block_toInclude': 9,
    'head_nonlinearity': 'GELU',
    'head_nonlinearity_kwargs': {},

    'lr': 1*10**-3,
    'penalty_orthogonality':0.00,
    'weight_decay': 0.0000,
    'gamma': 1-0.0000,
    'n_epochs': 9999999,
    'temperature': 0.1,
    'l2_alpha': 0.0000,
    
    'augmentation': {
        'Scale_image_sum': {'sum_val':1, 'epsilon':1e-9, 'min_sub':True},
        'AddPoissonNoise': {'scaler_bounds':(1.0*10**(3.5), 1.0*10**(4)), 'prob':0.7, 'base':1000, 'scaling':'log'},
        'Horizontal_stripe_scale': {'alpha_min_max':(0.5, 1), 'im_size':(36,36), 'prob':0.3},
        'Horizontal_stripe_shift': {'alpha_min_max':(1  , 2), 'im_size':(36,36), 'prob':0.3},
        'RandomHorizontalFlip': {'p':0.5},
        'RandomAffine': {
            'degrees':(-180,180),
            'translate':(0.1, 0.1), #0, .3, .45 (DEFAULT)
            'scale':(0.6, 1.2), # no scale (1,1), (0.4, 1.5)
            'shear':(-8, 8, -8, 8),
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
#             'img_size_out':[72,72],
            'img_size_out':[224,224],
        },
        'TileChannels': {'dim':0, 'n_channels':3},
    },
}



## make params dicts with grid swept values
params = copy.deepcopy(params_template)
params = [container_helpers.deep_update_dict(params, ['head_pool_method'], val) for val in ['AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']]
# params = container_helpers.flatten_list([[container_helpers.deep_update_dict(p, ['lr'], val) for val in [0.00001, 0.0001, 0.001]] for p in params])

params_unchanging, params_changing = container_helpers.find_differences_across_dictionaries(params)


## notes that will be saved as a text file
notes = \
"""
Testing out pooling method. AdaptiveMaxPool2d run should be redudant with the default in other runs.
"""

with open(str(Path(dir_save) / 'notes.txt'), mode='a') as f:
    f.write(notes)



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


## run batch_run function
paths_scripts = [path_script]
params_list = params
# sbatch_config_list = [sbatch_config]
max_n_jobs=3
name_save='jobNum_'


## define print log paths
paths_log = [str(Path(dir_save) / f'{name_save}{jobNum}' / 'print_log_%j.log') for jobNum in range(len(params))]

## define slurm SBATCH parameters
sbatch_config_list = \
[f"""#!/usr/bin/bash
#SBATCH --job-name=simCLR_pool1
#SBATCH --output={path}
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00

unset XDG_RUNTIME_DIR

cd /n/data1/hms/neurobio/sabatini/rich/

date

echo "loading modules"
module load gcc/9.2.0 cuda/11.2

echo "activating environment"
source activate ROI_env

echo "starting job"
python "$@"
""" for path in paths_log]

server.batch_run(paths_scripts=paths_scripts,
                    params_list=params_list,
                    sbatch_config_list=sbatch_config_list,
                    max_n_jobs=max_n_jobs,
                    dir_save=str(dir_save),
                    name_save=name_save,
                    verbose=True,
                    )
