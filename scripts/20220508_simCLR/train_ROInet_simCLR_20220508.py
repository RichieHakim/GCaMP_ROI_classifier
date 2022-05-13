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
import os
import copy
import pathlib
from pathlib import Path
import time
import gc
from functools import partial

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


# # dir_save = '/media/rich/bigSSD/analysis_data/ROI_net_training/testing_dispatcher_20220504'
# dir_save = Path(r'/media/rich/bigSSD/')

# # params_template = {
# params = {
#     'paths': {
#         'dir_github':'/media/rich/Home_Linux_partition/github_repos',
#         'fileName_save_model':'ConvNext_tiny_1.0unfrozen_simCLR',
#         'path_data_training':'/media/rich/bigSSD/analysis_data/ROIs_for_training/sf_sparse_36x36_20220503.npz',
#     },
    
#     'prefs': {
#         'saveModelIteratively':True,
#         'saveLogs':True,
#     },
    
#     'useGPU_training': True,
#     'useGPU_dataloader': False,
#     'dataloader_kwargs':{
#         'batch_size': 1024,
#         'shuffle': True,
#         'drop_last': True,
#         'pin_memory': True,
#         'num_workers': 18,
#         'persistent_workers': True,
#         'prefetch_factor': 2,
# #         'num_workers': 4,
# #         'persistent_workers': True,
# #         'prefetch_factor': 1,
#     },
#     'inner_batch_size': 256,

#     'torchvision_model': 'convnext_tiny',

#     'pre_head_fc_sizes': [256, 128],
#     'post_head_fc_sizes': [128],
#     'block_to_unfreeze': '6.0',
#     'n_block_toInclude': 9,
#     'head_nonlinearity': 'GELU',
    
#     'lr': 1*10**-2,
#     'penalty_orthogonality':0.05,
#     'weight_decay': 0.0000,
#     'gamma': 1-0.0000,
#     'n_epochs': 9999999,
#     'temperature': 0.1,
#     'l2_alpha': 0.0000,
    
#     'augmentation': {
#         'Scale_image_sum': {'sum_val':1, 'epsilon':1e-9, 'min_sub':True},
#         'AddPoissonNoise': {'scaler_bounds':(10**(3.5), 10**(4)), 'prob':0.7, 'base':1000, 'scaling':'log'},
#         'Horizontal_stripe_scale': {'alpha_min_max':(0.5, 1), 'im_size':(36,36), 'prob':0.3},
#         'Horizontal_stripe_shift': {'alpha_min_max':(1  , 2), 'im_size':(36,36), 'prob':0.3},
#         'RandomHorizontalFlip': {'p':0.5},
#         'RandomAffine': {
#             'degrees':(-180,180),
#             'translate':(0.1, 0.1), #0, .3, .45 (DEFAULT)
#             'scale':(0.6, 1.2), # no scale (1,1), (0.4, 1.5)
#             'shear':(-8, 8, -8, 8),
# #             'interpolation':torchvision.transforms.InterpolationMode.BILINEAR, 
#             'interpolation':'bilinear', 
#             'fill':0, 
#             'fillcolor':None, 
#             'resample':None,
#         },
#         'AddGaussianNoise': {'mean':0, 'std':0.0010, 'prob':0.5},
#         'ScaleDynamicRange': {'scaler_bounds':(0,1), 'epsilon':1e-9},
#         'WarpPoints': {
#             'r':[0.3, 0.6],
#             'cx':[-0.3, 0.3],
#             'cy':[-0.3, 0.3], 
#             'dx':[-0.24, 0.24], 
#             'dy':[-0.24, 0.24], 
#             'n_warps':2,
#             'prob':0.5,
#             'img_size_in':[36, 36],
# #             'img_size_out':[72,72],
#             'img_size_out':[224,224],
#         },
#         'TileChannels': {'dim':0, 'n_channels':3},
#     },
# }





### Import personal libraries

import sys

sys.path.append(params['paths']['dir_github'])
sys.path.append(str(Path(params['paths']['dir_github']) / 'GCaMP_ROI_classifier'))

# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import torch_helpers, path_helpers
from GCaMP_ROI_classifier import util, models, training, augmentation, dataset

def write_to_log(text, path_log, mode='a', start_on_new_line=True, pref_print=True, pref_save=True):
    if pref_print:
        print(text)
    if pref_save:
        with open(path_log, mode=mode) as log:
            if start_on_new_line==True:
                log.write('\n')
            log.write(text)
write_to_log = partial(write_to_log, pref_print=params['prefs']['log_print'], pref_save=params['prefs']['log_save'])




### Prepare paths

path_saveModel = str((dir_save / params['paths']['fileName_save_model']).with_suffix('.pth'))
path_saveLog = str(dir_save / 'log.txt')
path_saveLoss = (dir_save / 'loss.npy')

device_train = torch_helpers.set_device(use_GPU=params['useGPU_training'], verbose=False)



write_to_log(path_log=path_saveLog, text=f'sys.version: {sys.version_info}')
write_to_log(path_log=path_saveLog, text=f"sys.version: {os.environ['CONDA_DEFAULT_ENV']}")




### Import unlabeled training data

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  loading data...')

import scipy.sparse

sf_sparse = scipy.sparse.load_npz(params['paths']['path_data_training'])

sf_dense = torch.as_tensor(sf_sparse.toarray().reshape(sf_sparse.shape[0], 36,36), dtype=torch.float32)

##toss any NaNs

# print(f'Number of masks: {sf_dense.shape}')
ROIs_without_NaNs = ~torch.any(torch.any(torch.isnan(sf_dense), dim=1), dim=1)
ROIs_nonAllZero = (torch.max(torch.max(sf_dense, dim=1)[0], dim=1)[0] > 0)
ROIs_toKeep = torch.where(ROIs_without_NaNs * ROIs_nonAllZero)[0]
masks_cat = sf_dense[ROIs_toKeep]

n_masks_removed = np.sum(sf_dense.shape[0] - ROIs_toKeep.shape[0])
# print(f'Number of masks: {masks_cat.shape}')

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  data loaded.')




### Define augmentation pipeline

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  constructing augmentation pipeline and dataloader...')

transforms = torch.nn.Sequential(
    *[augmentation.__dict__[key](**params) for key,params in params['augmentation'].items()]
)
scripted_transforms = torch.jit.script(transforms)

device_dataloader = torch_helpers.set_device(use_GPU=params['useGPU_dataloader'])

dataset_train = dataset.dataset_simCLR(
    torch.as_tensor(masks_cat, device=device_dataloader, dtype=torch.float32), 
    torch.as_tensor(torch.zeros(masks_cat.shape[0]), device=device_dataloader, dtype=torch.float32),
    n_transforms=2,
    class_weights=np.array([1]),
    # class_weights=np.array([1]*4)[np.random.randint(0,4, X_train.shape[0])],
    transform=scripted_transforms,
    # DEVICE='cpu',
    DEVICE=device_dataloader,
    dtype_X=torch.float32,
    dtype_y=torch.int64,
    # temp_uncertainty=1
)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    **params['dataloader_kwargs']
)

# import matplotlib.pyplot as plt
# %matplotlib notebook

# idx_rand = np.random.randint(0,masks_cat.shape[0], 10)
# for ii in idx_rand:
#     fig, axs = plt.subplots(1,2)
#     # print(dataset_train[ii][0][0][0].shape)
#     axs[0].imshow(dataset_train[ii][0][0][0].cpu())
#     axs[1].imshow(dataset_train[ii][0][1][0].cpu())

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  augmentation pipeline and dataloader constructed.')





### Define ModelTackOn

class ModelTackOn(torch.nn.Module):
    def __init__(
        self, 
        base_model, 
        un_modified_model,
        data_dim=(1,3,36,36), 
        pre_head_fc_sizes=[100], 
        post_head_fc_sizes=[100], 
        classifier_fc_sizes=None, 
        nonlinearity='relu', 
        kwargs_nonlinearity={},
    ):
            super(ModelTackOn, self).__init__()
            self.base_model = base_model
            final_base_layer = list(un_modified_model.children())[-1]
            # final_base_layer = list(list(model.children())[-1].children())[-1]
            # print(final_base_layer)
            
            self.data_dim = data_dim

            self.pre_head_fc_lst = []
            self.post_head_fc_lst = []
            self.classifier_fc_lst = []
                
            self.nonlinearity = nonlinearity
            self.kwargs_nonlinearity = kwargs_nonlinearity

            self.init_prehead(final_base_layer, pre_head_fc_sizes)
            self.init_posthead(pre_head_fc_sizes[-1], post_head_fc_sizes)
            if classifier_fc_sizes is not None:
                self.init_classifier(pre_head_fc_sizes[-1], classifier_fc_sizes)
            
    def init_prehead(self, prv_layer, pre_head_fc_sizes):
        for i, pre_head_fc in enumerate(pre_head_fc_sizes):
            if i == 0:
#                 in_features = prv_layer.in_features if hasattr(prv_layer,'in_features') else 1280
#                 in_features = prv_layer.in_features if hasattr(prv_layer,'in_features') else 960
#                 in_features = prv_layer.in_features if hasattr(prv_layer,'in_features') else 768
#                 in_features = prv_layer.in_features if hasattr(prv_layer,'in_features') else 1536
#                 in_features = prv_layer.in_features if hasattr(prv_layer,'in_features') else 1024
                in_features = self.base_model(torch.rand(*(self.data_dim))).data.squeeze().shape[0]  ## RH EDIT
            else:
                in_features = pre_head_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=pre_head_fc)
            self.add_module(f'PreHead_{i}', fc_layer)
            self.pre_head_fc_lst.append(fc_layer)

#             if i < len(pre_head_fc_sizes) - 1:
#             non_linearity = torch.nn.ReLU()
#             non_linearity = torch.nn.GELU()
            non_linearity = torch.nn.__dict__[self.nonlinearity](**self.kwargs_nonlinearity)
            self.add_module(f'PreHead_{i}_NonLinearity', non_linearity)
            self.pre_head_fc_lst.append(non_linearity)

    def init_posthead(self, prv_size, post_head_fc_sizes):
        for i, post_head_fc in enumerate(post_head_fc_sizes):
            if i == 0:
                in_features = prv_size
            else:
                in_features = post_head_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=post_head_fc)
            self.add_module(f'PostHead_{i}', fc_layer)
            self.post_head_fc_lst.append(fc_layer)

#                 non_linearity = torch.nn.ReLU()
#                 non_linearity = torch.nn.GELU()
            non_linearity = torch.nn.__dict__[self.nonlinearity](**self.kwargs_nonlinearity)    
            self.add_module(f'PostHead_{i}_NonLinearity', non_linearity)
            self.pre_head_fc_lst.append(non_linearity)
    
    def init_classifier(self, prv_size, classifier_fc_sizes):
            for i, classifier_fc in enumerate(classifier_fc_sizes):
                if i == 0:
                    in_features = prv_size
                else:
                    in_features = classifier_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=classifier_fc)
            self.add_module(f'Classifier_{i}', fc_layer)
            self.classifier_fc_lst.append(fc_layer)

    def reinit_classifier(self):
        for i_layer, layer in enumerate(self.classifier_fc_lst):
            layer.reset_parameters()
    
#     def forward(self, X):
#         interim = self.base_model(X)
#         interim = self.get_head(interim)
#         interim = self.get_latent(interim)
#         return interim

    def forward_classifier(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.classify(interim)
        return interim

    def forward_latent(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.get_latent(interim)
        return interim


    def get_head(self, base_out):
        # print('base_out', base_out.shape)
        head = base_out
        for pre_head_layer in self.pre_head_fc_lst:
          # print('pre_head_layer', pre_head_layer.in_features)
          head = pre_head_layer(head)
          # print('head', head.shape)
        return head

    def get_latent(self, head):
        latent = head
        for post_head_layer in self.post_head_fc_lst:
            latent = post_head_layer(latent)
        return latent

    def classify(self, head):
        logit = head
        for classifier_layer in self.classifier_fc_lst:
            logit = classifier_layer(logit)
        return logit

    def set_pre_head_grad(self, requires_grad=True):
        for layer in self.pre_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad
                
    def set_post_head_grad(self, requires_grad=True):
        for layer in self.post_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def set_classifier_grad(self, requires_grad=True):
        for layer in self.classifier_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def prep_contrast(self):
        self.set_pre_head_grad(requires_grad=True)
        self.set_post_head_grad(requires_grad=True)
        self.set_classifier_grad(requires_grad=False)

    def prep_classifier(self):
        self.set_pre_head_grad(requires_grad=False)
        self.set_post_head_grad(requires_grad=False)
        self.set_classifier_grad(requires_grad=True)





### Import pretrained model

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  importing pretrained model...')

import torchvision.models

# base_model_frozen = torchvision.models.resnet101(pretrained=True)
# base_model_frozen = torchvision.models.resnet18(pretrained=True)
# base_model_frozen = torchvision.models.wide_resnet50_2(pretrained=True)
# base_model_frozen = torchvision.models.resnet50(pretrained=True)

# base_model_frozen = torchvision.models.efficientnet_b0(pretrained=True)

# base_model_frozen = torchvision.models.convnext_tiny(pretrained=True)
# base_model_frozen = torchvision.models.convnext_small(pretrained=True)
# base_model_frozen = torchvision.models.convnext_base(pretrained=True)
# base_model_frozen = torchvision.models.convnext_large(pretrained=True)


# base_model_frozen = torchvision.models.mobilenet_v3_large(pretrained=True)

base_model_frozen = torchvision.models.__dict__[params['torchvision_model']](pretrained=True)

for param in base_model_frozen.parameters():
    param.requires_grad = False

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  imported pretrained model')



### Make combined model

## Tacking on the latent layers needs to be done in a few steps.

## 0. Chop the base model
## 1. Tack on a pooling layer to reduce the size of the convlutional parameters
## 2. Determine the size of the output (internally done in ModelTackOn)
## 3. Tack on a linear layer of the correct size  (internally done in ModelTackOn)

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  making combined model...')

model_chopped = torch.nn.Sequential(list(base_model_frozen.children())[0][:params['n_block_toInclude']])  ## 0.
model_chopped_pooled = torch.nn.Sequential(model_chopped, torch.nn.__dict__[params['head_pool_method']](**params['head_pool_method_kwargs']), torch.nn.Flatten())  ## 1.

image_out_size = list(dataset_train[0][0][0].shape)
data_dim = tuple([1] + list(image_out_size))

## 2. , 3.
model = ModelTackOn(
#     model_chopped.to('cpu'),
    model_chopped_pooled.to('cpu'),
    base_model_frozen.to('cpu'),
    data_dim=data_dim,
    pre_head_fc_sizes=params['pre_head_fc_sizes'], 
    post_head_fc_sizes=params['post_head_fc_sizes'], 
    classifier_fc_sizes=None,
    nonlinearity=params['head_nonlinearity'],
    kwargs_nonlinearity=params['head_nonlinearity_kwargs'],
)
model.train();

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  made combined model')




### unfreeze particular blocks in model

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  unfreezing layers...')

mnp = [name for name, param in model.named_parameters()]  ## 'model named parameters'
mnp_blockNums = [name[name.find('.'):name.find('.')+8] for name in mnp]  ## pulls out the numbers just after the model name
mnp_nums = [path_helpers.get_nums_from_string(name) for name in mnp_blockNums]  ## converts them to numbers
block_to_freeze_nums = path_helpers.get_nums_from_string(params['block_to_unfreeze'])  ## converts the input parameter specifying the block to freeze into a number for comparison

m_baseName = mnp[0][:mnp[0].find('.')]

for ii, (name, param) in enumerate(model.named_parameters()):
    if m_baseName in name:
#         print(name)
        if mnp_nums[ii] < block_to_freeze_nums:
            param.requires_grad = False
        elif mnp_nums[ii] >= block_to_freeze_nums:
            param.requires_grad = True

names_layers_requiresGrad = [( param.requires_grad , name ) for name,param in list(model.named_parameters())]

# names_layers_requiresGrad

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  finished unfreezing layers')




### Save run outputs

## The training step is written so that it can run until a job ends, so this needs to be saved before

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  saving run outputs...')

run_outputs = {
    'dir_save': str(dir_save),    
    'path_save_runOutputs': str(dir_save / 'run_outputs.json'),    
    'path_saveModel': str(path_saveModel),
    'path_saveLog': str(path_saveLog),
    'path_saveLoss': str(path_saveLoss),
    'device_train': device_train,
    'masks_training_shape': list(masks_cat.shape),
    'n_masks_removed': int(n_masks_removed),
    'image_resized_shape': list(dataset_train[0][0][0].shape),
    'names_layers_requiresGrad': names_layers_requiresGrad,
}
run_outputs;

import json
with open(run_outputs['path_save_runOutputs'], 'w') as f:
    json.dump(run_outputs, f) 

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  saved run outputs')




### Training

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  preparing training...')


model.to(device_train)
model.prep_contrast()
model.forward = model.forward_latent

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

criterion = [CrossEntropyLoss()]
optimizer = Adam(
    model.parameters(), 
    lr=params['lr'],
#     lr=1*10**-2,
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                   gamma=params['gamma'],
#                                                    gamma=1,
                                                  )

criterion = [_.to(device_train) for _ in criterion]

write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  starting training...')


# model.load_state_dict(torch.load('/media/rich/bigSSD/ConvNext_tiny_1.pth'))

losses_train, losses_val = [], [np.nan]
for epoch in tqdm(range(params['n_epochs'])):
    write_to_log(path_log=path_saveLog, text=f'in epoch loop')

    print(f'epoch: {epoch}')
    
    losses_train = training.epoch_step(
        dataloader_train, 
        model, 
        optimizer, 
        criterion,
        scheduler=scheduler,
        temperature=params['temperature'],
        # l2_alpha,
        penalty_orthogonality=params['penalty_orthogonality'],
        mode='semi-supervised',
        loss_rolling_train=losses_train, 
        loss_rolling_val=losses_val,
        device=device_train, 
        inner_batch_size=params['inner_batch_size'],
        verbose=2,
        verbose_update_period=1,
        log_function=partial(write_to_log, path_log=path_saveLog),

#                                     do_validation=False,
#                                     X_val=x_feed_through_val,
#                                     y_val=torch.as_tensor(y_val, device=DEVICE)
)
    
    ## save loss stuff
    write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}, completed epoch: {epoch}, loss: {losses_train[-1]}, lr: {scheduler.get_last_lr()[0]}')
    if params['prefs']['saveLogs']:
        np.save(path_saveLoss, losses_train)
    
    ## if loss becomes NaNs, don't save the network and stop training
    if torch.isnan(torch.as_tensor(losses_train[-1])):
        write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}, EXITED DUE TO loss==NaN')
        break
        
    ## save model
    if params['prefs']['saveModelIteratively']:
        torch.save(model.state_dict(), path_saveModel)