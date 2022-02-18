
import sys
import torch
import numpy as np
import importlib
from tqdm.notebook import tqdm
import pickle
from basic_neural_processing_modules import h5_handling, pickle_helpers

def load_h5(path, h5_keys):
    base_data = h5_handling.simple_load(path=path)
    h5_subsets = [base_data[key] for key in h5_keys]
    return torch.as_tensor(np.concatenate(h5_subsets, axis=0), dtype=torch.float32, device='cpu')

def drop_nan_imgs(rois):
    ROIs_without_NaNs = torch.where(~torch.any(torch.any(torch.isnan(rois), dim=1), dim=1))[0]
    return rois[ROIs_without_NaNs]


## TODO: Troubleshoot the runtime on this
def dataloader_to_latents(dataloader, model, DEVICE='cpu'):
    def subset_to_latents(data):
        return model.get_head(model.base_model(data[0][0].to(DEVICE))).detach().cpu()
    return torch.cat([subset_to_latents(data) for data in tqdm(dataloader)], dim=0)

def load_classifier_model(classifier_name):
    with open(classifier_name, 'rb') as classifier_model_file:
        classifier = pickle.load(classifier_model_file)
    return classifier

from GCaMP_ROI_classifier.new_stuff import util
def get_returns(latents, classifier_model, path_to_model, path_to_classifier):
    ret = {}
    ret['path_to_model'] = path_to_model
    ret['path_to_classifier'] = path_to_classifier
    ret['latents'] = latents
    ret['proba'] = classifier_model.predict_proba(latents)
    ret['preds'] = np.argmax(ret['proba'], axis=-1)
    ret['uncertainty'] = util.loss_uncertainty(torch.as_tensor(ret['proba']), temperature=1, class_value=None).detach().cpu().numpy()
    
    params = classifier_model.get_params()
    ret['params'] = params
    return ret

