import numpy as np

import torch
import torch.cuda
from torch.autograd import Variable

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import time


def train_step( X_train_batch, y_train_batch, 
                model, optimizer, criterion, scheduler, 
                ):

        optimizer.zero_grad()

        features = model(X_train_batch)
        
        logits, labels = info_nce_loss(features, batch_size=X_train_batch.shape[0]/2, n_views=2, temperature=0.04, DEVICE=X_train_batch.device)
        loss_train = criterion(logits, labels)
        # loss_train = criterion(y_hat, y_train_batch)

        loss_train.backward()
        optimizer.step()
        scheduler.step()

        return loss_train.item()

def epoch_step( dataloader, 
                model, 
                optimizer, 
                criterion, 
                scheduler=None, 
                loss_rolling_train=[], 
                device='cpu', 
                do_validation=False,
                validation_Object=None,
                loss_rolling_val=[],
                verbose=False,
                verbose_update_period=100
                ):

    def print_info(batch, n_batches, loss_train, loss_val, learning_rate, precis=5):
        print(f'Iter: {batch}/{n_batches}, loss_train: {loss_train:.{precis}}, loss_val: {loss_val:.{precis}}, lr: {learning_rate:.{precis}}')

    for i_batch, (X_batch, y_batch, idx_batch) in enumerate(dataloader):
        X_batch = torch.cat(X_batch, dim=0)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        for i_transform in range(X_batch.shape[1]): # X_batch.shape[1] = number of transforms
            # loss = train_step(X_batch[:,i_transform,...], y_batch, model, optimizer, criterion, scheduler)
            # X_batch = X_batch.reshape([X_batch.shape[0]*X_batch.shape[1]] + list(X_batch.shape[2:]))
            loss = train_step(X_batch, y_batch, model, optimizer, criterion, scheduler)
            loss_rolling_train.append(loss)
            if do_validation:
                loss = validation_Object.get_predictions()
                loss_rolling_val.append(loss)
        if verbose>0:
            if i_batch%verbose_update_period == 0:
                print_info( batch=i_batch,
                            n_batches=len( dataloader),
                            loss_train=loss_rolling_train[-1],
                            loss_val=loss_rolling_val[-1],
                            learning_rate=scheduler.get_last_lr()[0],
                            precis=5)
    return loss_rolling_train

class validation_Obj():
    def __init__(   self, 
                    X_val, 
                    y_val, 
                    model,
                    criterion,
                    DEVICE='cpu',
                    dtype_X=torch.float32,
                    dtype_y=torch.int64):

        self.X_val = torch.as_tensor(X_val, dtype=dtype_X, device=DEVICE)[:,None,...] # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.y_val = torch.as_tensor(y_val, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.
        self.model = model
        self.criterion = criterion

    # def get_predictions(self):
    #     with torch.no_grad():
    #         ### NOT IMPLEMENTED YET ###
    #         self.X_val_transformed = 
    #         features = self.model(self.X_val)
        
    #         logits, labels = info_nce_loss(features)
    #         loss = self.criterion(logits, labels)

    #         return loss.item()

# # from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
def info_nce_loss(features, batch_size, n_views=2, temperature=0.5, DEVICE='cpu'):

    # make (double) diagonal matrix
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(DEVICE)

    # normalize to unit hypersphere
    features = torch.nn.functional.normalize(features, dim=1)

    # compute (double) covariance matrix
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) # logits column 1 is positives, the rest of the columns are negatives
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE) # all labels are 0 because first column in logits is positives

    logits = logits / temperature
    return logits, labels
