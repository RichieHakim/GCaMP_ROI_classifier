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

        y_hat = model(X_train_batch)
        loss_train = criterion(y_hat, y_train_batch)

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
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
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

    def get_predictions(self):
        with torch.no_grad():
            y_hat = self.model(self.X_val)
            loss = self.criterion(y_hat, self.y_val)
            return loss.item(), y_hat