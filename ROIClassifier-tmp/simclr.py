# # Imports
# import torch
# import torchvision
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# import os, sys, importlib
# from io import BytesIO
# from urllib.request import urlopen
# from zipfile import ZipFile

# REPO_PATH = "neuromatch_ssl_tutorial"

# # @markdown Import modules designed for use in this tutorials
# from neuromatch_ssl_tutorial.modules import data, load, models, plot_util
# from neuromatch_ssl_tutorial.modules import data, load, models, plot_util

# import ipywidgets as widgets       # interactive display
# import warnings
# from IPython.display import display, Image # to visualize images

# # Call `set_seed` function in the exercises to ensure reproducibility.
# import random
# import torch



# # @markdown Function to set test custom contrastive loss function: `test_custom_contrastive_loss_fct()`
# def test_custom_contrastive_loss_fct(custom_simclr_contrastive_loss):
#   rand_proj_feat1 = torch.rand(100, 1000)
#   rand_proj_feat2 = torch.rand(100, 1000)
#   loss_custom = custom_simclr_contrastive_loss(rand_proj_feat1, rand_proj_feat2)
#   loss_ground_truth = models.contrastive_loss(rand_proj_feat1,rand_proj_feat2)

#   if torch.allclose(loss_custom, loss_ground_truth):
#     print("custom_simclr_contrastive_loss() is correctly implemented.")
#   else:
#     print("custom_simclr_contrastive_loss() is NOT correctly implemented.")


# #@title Set random seed

# #@markdown Executing `set_seed(seed=seed)` you are setting the seed

# # for DL its critical to set the random seed so that students can have a
# # baseline to compare their results to expected results.
# # Read more here: https://pytorch.org/docs/stable/notes/randomness.html

# def set_seed(seed=None, seed_torch=True):
#   if seed is None:
#     seed = np.random.choice(2 ** 32)
#   random.seed(seed)
#   np.random.seed(seed)
#   if seed_torch:
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

#   print(f'Random seed {seed} has been set.')


# # In case that `DataLoader` is used
# def seed_worker(worker_id):
#   worker_seed = torch.initial_seed() % 2**32
#   np.random.seed(worker_seed)
#   random.seed(worker_seed)






# #@title Set device (GPU or CPU). Execute `set_device()`
# # especially if torch modules used.

# # inform the user if the notebook uses GPU or CPU.

# def set_device():
#   device = "cuda" if torch.cuda.is_available() else "cpu"
#   if device != "cuda":
#     print("WARNING: For this notebook to perform best, "
#         "if possible, in the menu under `Runtime` -> "
#         "`Change runtime type.`  select `GPU` ")
#   else:
#     print("GPU is enabled in this notebook.")

#   return device


# # Set global variables
# SEED = 2021
# set_seed(seed=SEED)
# DEVICE = set_device()


# def nothing():
#   # @markdown ### Pre-load variables (allows each section to be run independently)

#   # Section 1
#   dSprites = data.dSpritesDataset(
#       os.path.join(REPO_PATH, "dsprites", "dsprites_subset.npz")
#       )

#   dSprites_torchdataset = data.dSpritesTorchDataset(
#     dSprites,
#     target_latent="shape"
#     )

#   train_sampler, test_sampler = data.train_test_split_idx(
#     dSprites_torchdataset,
#     fraction_train=0.8,
#     randst=SEED
#     )

#   supervised_encoder = load.load_encoder(REPO_PATH,
#                                         model_type="supervised",
#                                         verbose=False)

#   # Section 2
#   custom_torch_RSM_fct = None  # default is used instead

#   # Section 3
#   random_encoder = load.load_encoder(REPO_PATH,
#                                     model_type="random",
#                                     verbose=False)

#   # Section 4
#   vae_encoder = load.load_encoder(REPO_PATH,
#                                   model_type="vae",
#                                   verbose=False)

#   # Section 5
#   invariance_transforms = torchvision.transforms.RandomAffine(
#       degrees=90,
#       translate=(0.2, 0.2),
#       scale=(0.8, 1.2)
#       )
#   dSprites_invariance_torchdataset = data.dSpritesTorchDataset(
#       dSprites,
#       target_latent="shape",
#       simclr=True,
#       simclr_transforms=invariance_transforms
#       )

#   # Section 6
#   simclr_encoder = load.load_encoder(REPO_PATH,
#                                     model_type="simclr",
#                                     verbose=False)













#   # call this before any dataset/network initializing or training,
#   # to ensure reproducibility
#   set_seed(SEED)

#   # DEMO: Try some random affine data augmentations combinations to apply to the images
#   invariance_transforms = torchvision.transforms.RandomAffine(
#       degrees=90,
#       translate=(0.2, 0.2),  # (in x, in y)
#       scale=(0.8, 1.2)   # min to max scaling
#       )

#   # initialize a simclr-specific torch dataset
#   dSprites_invariance_torchdataset = data.dSpritesTorchDataset(
#       dSprites,
#       target_latent="shape",
#       simclr=True,
#       simclr_transforms=invariance_transforms
#       )

#   # show a few example of pairs of image augmentations
#   _ = dSprites_invariance_torchdataset.show_images(randst=SEED)

# def custom_simclr_contrastive_loss(proj_feat1, proj_feat2, temperature=0.5):

#   """
#   custom_simclr_contrastive_loss(proj_feat1, proj_feat2)
#   Returns contrastive loss, given sets of projected features, with positive
#   pairs matched along the batch dimension.
#   Required args:
#   - proj_feat1 (2D torch Tensor): projected features for first image
#       augmentations (batch_size x feat_size)
#   - proj_feat2 (2D torch Tensor): projected features for second image
#       augmentations (batch_size x feat_size)

#   Optional args:
#   - temperature (float): relaxation temperature. (default: 0.5)
#   Returns:
#   - loss (float): mean contrastive loss
#   """
#   device = proj_feat1.device

#   if len(proj_feat1) != len(proj_feat2):
#     raise ValueError(f"Batch dimension of proj_feat1 ({len(proj_feat1)}) "
#                      f"and proj_feat2 ({len(proj_feat2)}) should be same")

#   batch_size = len(proj_feat1) # N
#   z1 = torch.nn.functional.normalize(proj_feat1, dim=1)
#   z2 = torch.nn.functional.normalize(proj_feat2, dim=1)

#   proj_features = torch.cat([z1, z2], dim=0) # 2N x projected feature dimension
#   similarity_matrix = torch.nn.functional.cosine_similarity(
#       proj_features.unsqueeze(1), proj_features.unsqueeze(0), dim=2
#       ) # dim: 2N x 2N

#   # initialize arrays to identify sets of positive and negative examples, of
#   # shape (batch_size * 2, batch_size * 2), and where
#   # 0 indicates that 2 images are NOT a pair (either positive or negative, depending on the indicator type)
#   # 1 indices that 2 images ARE a pair (either positive or negative, depending on the indicator type)
#   pos_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1).to(device)
#   neg_sample_indicators = (torch.ones(2 * batch_size) - torch.eye(2 * batch_size)).to(device)

#   #################################################
#   # Fill in missing code below (...),
#   # then remove or comment the line below to test your function
#   # raise NotImplementedError("Exercise: Implement SimCLR loss.")
#   #################################################
#   # EXERCISE: Implement the SimClr loss calculation
#   # Calculate the numerator of the Loss expression by selecting the appropriate elements from similarity_matrix.
#   # Use the pos_sample_indicators tensor
#   numerator = torch.exp(similarity_matrix/temperature)[pos_sample_indicators.bool()]

#   # Calculate the denominator of the Loss expression by selecting the appropriate elements from similarity_matrix,
#   # and summing over pairs for each item.
#   # Use the neg_sample_indicators tensor
#   denominator = torch.sum(neg_sample_indicators * torch.exp(similarity_matrix/temperature), dim=1)

#   if (denominator < 1e-8).any(): # clamp to avoid division by 0
#     denominator = torch.clamp(denominator, 1e-8)

#   loss = torch.mean(-torch.log(numerator / denominator))

#   return loss



# # Uncomment below to test your function
# test_custom_contrastive_loss_fct(custom_simclr_contrastive_loss)

# # call this before any dataset/network initializing or training,
# # to ensure reproducibility
# set_seed(SEED)

# # Train SimCLR for a few epochs
# print("Training a SimCLR encoder with the custom contrastive loss...")
# num_epochs = 5
# _, test_simclr_loss_array = models.train_simclr(
#     encoder=models.EncoderCore(),
#     dataset=dSprites_invariance_torchdataset,
#     train_sampler=train_sampler,
#     num_epochs=num_epochs,
#     loss_fct=custom_simclr_contrastive_loss
#     )

# # Plot SimCLR loss over a few epochs.
# fig, ax = plt.subplots()
# ax.plot(test_simclr_loss_array)
# ax.set_title("SimCLR network loss")
# ax.set_xlabel("Epoch number")
# _ = ax.set_ylabel("Training loss")

# # Load SimCLR encoder pre-trained on the contrastive loss
# simclr_encoder = load.load_encoder(REPO_PATH, model_type="simclr")

# # call this before any dataset/network initializing or training,
# # to ensure reproducibility
# set_seed(SEED)

# print("Training a classifier on the pre-trained SimCLR encoder representations...")
# _, simclr_loss_array, _, _ = models.train_classifier(
#     encoder=simclr_encoder,
#     dataset=dSprites_torchdataset,
#     train_sampler=train_sampler,
#     test_sampler=test_sampler,
#     freeze_features=True, # keep the encoder frozen while training the classifier
#     num_epochs=10, # DEMO: Try different numbers of epochs
#     verbose=True
#     )

# fig, ax = plt.subplots()
# ax.plot(simclr_loss_array)
# ax.set_title("Loss of classifier trained on a SimCLR encoder.")
# ax.set_xlabel("Epoch number")
# _ = ax.set_ylabel("Training loss")





import numpy as np
import copy

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import logging
import os
import sys


import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)




# Source: https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        # print('similarity_matrix', similarity_matrix.shape)
        # print('mask', mask.shape)
        # print('labels', labels.shape)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                # print(images.shape)
                # images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        # save_checkpoint({
        #     'epoch': self.args.epochs,
        #     'arch': self.args.arch,
        #     'state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        # }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

class WindowedDataset(Dataset):
    def __init__(self, X_untiled, y_input, transform=None, target_transform=None):
        self.X_untiled = X_untiled # first dim will be subsampled from
        self.y_input = y_input # first dim will be subsampled from
        # self.win_range = win_range
        self.n_samples = y_input.shape[0]
        # self.usable_idx = torch.arange(-self.win_range[0] , self.n_samples-self.win_range[1]+1)
        
        if X_untiled.shape[0] != y_input.shape[0]:
            raise ValueError('RH: X and y must have same first dimension shape')

    def __len__(self):
        return self.n_samples
    
    # def check_bound_errors(self, idx):
    #     idx_toRemove = []
    #     for val in idx:
    #         if (val+self.win_range[0] < 0) or (val+self.win_range[1] > self.n_samples):
    #             idx_toRemove.append(val)
    #     if len(idx_toRemove) > 0:
    #         raise ValueError(f'RH: input idx is too close to edges. Remove idx: {idx_toRemove}')

    def __getitem__(self, idx):
#         print(idx)
#         self.check_bound_errors(idx)
        X_subset_tiled = self.X_untiled[idx]
        y_subset = self.y_input[idx]
        return X_subset_tiled, y_subset

def make_WindowedDataloader(X, y, batch_size=64, drop_last=True, **kwargs_dataloader):
    dataset = WindowedDataset(X, y)

    # sampler = torch.utils.data.SubsetRandomSampler(dataset.usable_idx, generator=None)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=None, generator=None)
    
    
    if kwargs_dataloader is None:
        kwargs_dataloader = {'shuffle': False,
                             'pin_memory': False,
                             'num_workers':0
                            }
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=drop_last,
                            sampler=sampler,
                            **kwargs_dataloader,
                            )
    # dataloader.sample_shape = [dataloader.batch_size] + list(dataset[-win_range[0]][0].shape)
    return dataloader, dataset, sampler



from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Dropout, Linear, Module

dropout_prob = 0.4
momentum_val = 0.9

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0),
            ReLU(),
            
            Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=0),
            MaxPool2d(kernel_size=2, stride=2),           
            ReLU(),
            Dropout(dropout_prob*1),
            
            Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),           
            ReLU(),
            Dropout(dropout_prob*1),

            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            MaxPool2d(kernel_size=2, stride=2),           
            ReLU(),
            Dropout(dropout_prob*1),   
        )

        self.linear_layers = Sequential(
            Linear(in_features=64, out_features=256),
            ReLU(),
            Dropout(dropout_prob*1),
            Linear(in_features=256, out_features=6),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.get_embedding(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x
    
    def get_embedding(self, x):
        x = x.float()
        x = self.cnn_layers(x)
        return x


















































# """ResNet in PyTorch.
# ImageNet-Style ResNet
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# Adapted from: https://github.com/bearpaw/pytorch-classification
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, is_last=False):
#         super(BasicBlock, self).__init__()
#         self.is_last = is_last
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         preact = out
#         out = F.relu(out)
#         if self.is_last:
#             return out, preact
#         else:
#             return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1, is_last=False):
#         super(Bottleneck, self).__init__()
#         self.is_last = is_last
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         preact = out
#         out = F.relu(out)
#         if self.is_last:
#             return out, preact
#         else:
#             return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves
#         # like an identity. This improves the model by 0.2~0.3% according to:
#         # https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for i in range(num_blocks):
#             stride = strides[i]
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, layer=100):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         return out


# def resnet18(**kwargs):
#     return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# def resnet34(**kwargs):
#     return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def resnet50(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet101(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


# model_dict = {
#     'resnet18': [resnet18, 512],
#     'resnet34': [resnet34, 512],
#     'resnet50': [resnet50, 2048],
#     'resnet101': [resnet101, 2048],
# }


# class LinearBatchNorm(nn.Module):
#     """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
#     def __init__(self, dim, affine=True):
#         super(LinearBatchNorm, self).__init__()
#         self.dim = dim
#         self.bn = nn.BatchNorm2d(dim, affine=affine)

#     def forward(self, x):
#         x = x.view(-1, self.dim, 1, 1)
#         x = self.bn(x)
#         x = x.view(-1, self.dim)
#         return x


# class SupConResNet(nn.Module):
#     """backbone + projection head"""
#     def __init__(self, name='resnet50', head='mlp', feat_dim=128):
#         super(SupConResNet, self).__init__()
#         model_fun, dim_in = model_dict[name]
#         self.encoder = model_fun()
#         if head == 'linear':
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(dim_in, dim_in),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(dim_in, feat_dim)
#             )
#         else:
#             raise NotImplementedError(
#                 'head not supported: {}'.format(head))

#     def forward(self, x):
#         feat = self.encoder(x)
#         feat = F.normalize(self.head(feat), dim=1)
#         return feat


# class SupCEResNet(nn.Module):
#     """encoder + classifier"""
#     def __init__(self, name='resnet50', num_classes=10):
#         super(SupCEResNet, self).__init__()
#         model_fun, dim_in = model_dict[name]
#         self.encoder = model_fun()
#         self.fc = nn.Linear(dim_in, num_classes)

#     def forward(self, x):
#         return self.fc(self.encoder(x))


# class LinearClassifier(nn.Module):
#     """Linear classifier"""
#     def __init__(self, name='resnet50', num_classes=10):
#         super(LinearClassifier, self).__init__()
#         _, feat_dim = model_dict[name]
#         self.fc = nn.Linear(feat_dim, num_classes)

#     def forward(self, features):
#         return self.fc(features)