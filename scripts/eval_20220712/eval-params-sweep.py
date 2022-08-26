### Import personal libraries

import importlib.util
import glob
from pathlib import Path
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import scipy.stats
from kymatio import Scattering2D
import json
import numpy as np
print(np.__version__)
import torchvision
print(torchvision.__version__)
import torch
from tqdm import tqdm, trange
import sklearn
from sklearn.model_selection import train_test_split, ShuffleSplit
import sys
import scipy.signal
import pickle

import os
print(f"script environment: {os.environ['CONDA_DEFAULT_ENV']}")

import sys
path_script, path_params, dir_save = sys.argv
# path_script = '/n/data1/hms/neurobio/sabatini/josh/github_repos/GCaMP_ROI_classifier/scripts/eval/eval-params-sweep.py'
# path_params = '/n/data1/hms/neurobio/sabatini/josh/github_repos/GCaMP_ROI_classifier/scripts/eval/params.json'
# dir_save = '/n/data1/hms/neurobio/sabatini/josh/github_repos/GCaMP_ROI_classifier/scripts/eval/save_dir/job_0'
output_job_dir = dir_save

import json
with open(path_params, 'r') as f:
    params_eval = json.load(f)

model_loc = params_eval['model_loc']

import shutil
from pathlib import Path
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));

sys.path.append(params_eval['github_loc'])
from basic_neural_processing_modules import torch_helpers, math_functions, classification, h5_handling, plotting_helpers, indexing, misc, decomposition, path_helpers
from GCaMP_ROI_classifier import util, models, training, augmentation, dataset

debug_mode = params_eval['debug_mode']
input_dir = params_eval['input_dir']
output_job_dir = dir_save
path_stat = params_eval['path_stat'] 
path_labels = params_eval['path_labels']


if debug_mode:
#     plot_title_params_key = 'temperature'
    classifier_n_splits = 2
    logistic_max_iter = 6000
    pc_sweep = [0, 10]
else:
#     plot_title_params_key = 'temperature'
    classifier_n_splits = params_eval['classifier_n_splits']
    logistic_max_iter = params_eval['logistic_max_iter']
    pc_sweep = params_eval['pc_sweep']

jns = sorted(glob.glob(str(Path(input_dir) / 'jobNum_*')))

if debug_mode:
    jns = jns[:2]

print('params_eval', params_eval)
print('input_dir', input_dir)

json_files = sorted(glob.glob(str(Path(input_dir) / '*.json')))
print(json_files)
if len(json_files) == 0:
    sys.exit(f'No jobs found. Returning. {json_files}')

with open(json_files[0], 'rb') as f:
    params_batch = json.load(f)
plot_title_job_num_ids = params_batch['params_changing']


def model_from_params(params, ModelTackOn, image_out_size=[3, 224, 224], pref_log_all_steps=False):
    base_model_frozen = torchvision.models.__dict__[params['torchvision_model']](pretrained=True)
    for param in base_model_frozen.parameters():
        param.requires_grad = False
    if pref_log_all_steps:
        write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  imported pretrained model')

    ### Make combined model

    ## Tacking on the latent layers needs to be done in a few steps.

    ## 0. Chop the base model
    ## 1. Tack on a pooling layer to reduce the size of the convlutional parameters
    ## 2. Determine the size of the output (internally done in ModelTackOn)
    ## 3. Tack on a linear layer of the correct size  (internally done in ModelTackOn)

    if pref_log_all_steps:
        write_to_log(path_log=path_saveLog, text=f'time:{time.ctime()}  making combined model...')

    model_chopped = torch.nn.Sequential(list(base_model_frozen.children())[0][:params['n_block_toInclude']])  ## 0.
    model_chopped_pooled = torch.nn.Sequential(model_chopped, torch.nn.AdaptiveAvgPool2d(output_size=1), torch.nn.Flatten())  ## 1.
    
    data_dim = tuple([1] + list(image_out_size))
    
    model = ModelTackOn(
    #     model_chopped.to('cpu'),
        model_chopped_pooled.to('cpu'),
        base_model_frozen.to('cpu'),
        data_dim=data_dim,
        pre_head_fc_sizes=params['pre_head_fc_sizes'], 
        post_head_fc_sizes=params['post_head_fc_sizes'], 
        classifier_fc_sizes=None,
        nonlinearity=params['head_nonlinearity'],
        kwargs_nonlinearity={},
    )
    return base_model_frozen, model_chopped, model_chopped_pooled, image_out_size, data_dim, model

def helper_make_dataset(X, scripted_transforms_classifier):
    out = dataset.dataset_simCLR(
        X=torch.as_tensor(X, device='cpu', dtype=torch.float32),
        y=torch.as_tensor(torch.zeros(X.shape[0]), device='cpu', dtype=torch.float32),
        n_transforms=1,
        class_weights=np.array([1]),
        transform=scripted_transforms_classifier,
        DEVICE='cpu',
        dtype_X=torch.float32,
    )
    return out

def helper_make_dataloader(ds):
    out = torch.utils.data.DataLoader( 
        ds,
#         batch_size=128,
        batch_size=8,
        shuffle=False,
        drop_last=False
    )
    return out

def get_balanced_sample_weights(labels):
    labels = np.int64(labels.copy())
    counts, vals = np.histogram(labels, bins=np.concatenate((np.unique(labels), [labels.max()+1])))
    vals = vals[:-1]

    n_labels = len(labels)
    weights = n_labels / counts
    
    sample_weights = np.array([weights[l] for l in labels])
    
    return sample_weights

def get_latents_swt(sfs, swt, device_model):
    sfs = torch.as_tensor(np.ascontiguousarray(sfs[None,...]), device=device_model, dtype=torch.float32)
    latents_swt = swt(sfs[None,...]).squeeze()
    latents_swt = latents_swt.reshape(latents_swt.shape[0], -1)
    return latents_swt

pth_fn_dct = {}
log_fn_dct = {}
loss_fn_dct = {}
params_fn_dct = {}
run_outputs_fn_dct = {}
sbatch_config_fn_dct = {}
base_py_fn_dct = {}

params_dct = {}
run_outputs_dct = {}

for jn in jns:
    print(jn)
    
    pth_fn = glob.glob(str(Path(jn) / '*.pth'))
    print('pth_fn', pth_fn)
    
    log_fn = glob.glob(str(Path(jn) / 'log.txt'))
    loss_fn = glob.glob(str(Path(jn) / 'loss.npy'))
    params_fn = glob.glob(str(Path(jn) / 'params.json'))
    run_outputs_fn = glob.glob(str(Path(jn) / 'run_outputs.json'))
    sbatch_config_fn = glob.glob(str(Path(jn) / 'sbatch_config.sh'))
    base_py_fn = glob.glob(str(Path(jn) / '*.py'))
    
    try:
        assert len(pth_fn) == 1
        assert len(log_fn) == 1
        assert len(loss_fn) == 1
        assert len(params_fn) == 1
        assert len(run_outputs_fn) == 1
        assert len(sbatch_config_fn) == 1
        assert len(base_py_fn) == 1
    except:
        print('Incorrect number of filenames detected... Continuing...')
        continue
    
    with open(params_fn[0], 'rb') as f:
        params_dct[jn] = json.load(f)
        
    with open(run_outputs_fn[0], 'rb') as f:
        run_outputs_dct[jn] = json.load(f)
    
    pth_fn_dct[jn] = pth_fn[0]
    log_fn_dct[jn] = log_fn[0]
    loss_fn_dct[jn] = loss_fn[0]
    params_fn_dct[jn] = params_fn[0]
    run_outputs_fn_dct[jn] = run_outputs_fn[0]
    sbatch_config_fn_dct[jn] = sbatch_config_fn[0]
    base_py_fn_dct[jn] = base_py_fn[0]


images_labeled = \
    util.import_multiple_stat_files(   
        paths_statFiles=[path_stat],
        out_height_width=[36,36],
        max_footprint_width=241,
        plot_pref=True
    )
# )


sf_all_cat = np.concatenate(images_labeled, axis=0)
sf_all_cat = np.concatenate((sf_all_cat, sf_all_cat), axis=0)

sf_ptiles = np.array([np.percentile(np.sum(sf>0, axis=(1,2)), 90) for sf in tqdm(images_labeled)])
scales_forRS = (250/sf_ptiles)**0.6
sf_rs = [np.stack([util.resize_affine(img, scale=scales_forRS[ii], clamp_range=True) for img in sf], axis=0) for ii, sf in enumerate(tqdm(images_labeled))]
images_labeled = np.concatenate(sf_rs, axis=0)
images_labeled = np.concatenate((images_labeled, images_labeled), axis=0)


labels = classification.squeeze_integers(np.concatenate([np.load(path) for path in path_labels]))

assert images_labeled.shape[0] == labels.shape[0] , 'num images in stat files does not correspond to num labels'


idx_toKeep = np.where(np.logical_not(labels == 4))[0]

images_labeled_clean = images_labeled[idx_toKeep]
labels_clean = labels[idx_toKeep]

spec = importlib.util.spec_from_file_location(model_loc.split('/')[-1].replace('.py', ''), params_eval['model_loc'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

mod_dct_dct = {}

scree_plt_dct = {}
corr_plt_dct = {}
acc_plt_dct = {}
cm_plt_dct = {}
embeddings_plt_dct = {}
title_val_plt_dct = {}
features_nn_plt_dct = {}
img_grid_plt_dct = {}
scores_nn_plt_dct = {}
labels_clean_plt_dct = {}
run_id_dct = {}
pc_id_dct = {}

i = -1

fitted_models_dct = {}

for jn in (pbar_params_set := tqdm(params_dct)):
    i += 1
    
    pbar_params_set.set_description(f'Params # — {i}')
    
    scree_plt_dct[jn] = {}
    corr_plt_dct[jn] = {}
    acc_plt_dct[jn] = {}
    cm_plt_dct[jn] = {}
    embeddings_plt_dct[jn] = {}
    title_val_plt_dct[jn] = {}
    features_nn_plt_dct[jn] = {}
    img_grid_plt_dct[jn] = {}
    scores_nn_plt_dct[jn] = {}
    labels_clean_plt_dct[jn] = {}
    
    fitted_models_dct[jn] = {}
    
    
    run_id_dct[jn] = str(plot_title_job_num_ids[i])
    print(run_id_dct)
    
    params = params_dct[jn]
    run_outputs = run_outputs_dct[jn]
    base_py_fn = base_py_fn_dct[jn]
    
    device_dataloader = torch_helpers.set_device(use_GPU=params['useGPU_dataloader'])
    DEVICE = torch_helpers.set_device(use_GPU=params['useGPU_training'])
    
    (base_model_frozen, model_chopped, model_chopped_pooled,
    image_out_size, data_dim, model) = model_from_params(params,
                                                         module.ModelTackOn,
                                                         image_out_size=run_outputs['image_resized_shape'],
                                                         pref_log_all_steps=False)
    
    
    
    
    mod_dct_dct[jn] = {name: param.detach().numpy() for name, param in model.named_parameters()}
    
    ### unfreeze particular blocks in model

    mnp = [name for name, param in model.named_parameters()]  ## 'model named parameters'
    mnp_blockNums = [name[name.find('.'):name.find('.')+8] for name in mnp]  ## pulls out the numbers just after the model name
    mnp_nums = [path_helpers.get_nums_from_string(name) for name in mnp_blockNums]  ## converts them to numbers
    
    m_baseName = mnp[0][:mnp[0].find('.')]
    
    ### Training
    
    model.forward = model.forward_latent
    
    transforms_classifier = torch.nn.Sequential(
        augmentation.ScaleDynamicRange(scaler_bounds=(0,1)),
        torchvision.transforms.Resize(
            size=(224, 224),   
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR), 
        augmentation.TileChannels(dim=0, n_channels=3),
    )

    scripted_transforms_classifier = torch.jit.script(transforms_classifier)
    
    
    
    dataset_labeled_clean = helper_make_dataset(images_labeled_clean, scripted_transforms_classifier)
    dataloader_labeled_clean = helper_make_dataloader(dataset_labeled_clean)

    model.eval()
    model.to(DEVICE)
    features_nn = torch.cat([model.get_head(model.base_model(data[0][0].to(DEVICE))).detach().cpu() for idata, data in enumerate(tqdm(dataloader_labeled_clean))], dim=0)
    
#     if i != 0:
#         print('All Close:', np.allclose(features_nn.numpy(), prv_features_nn))
    prv_features_nn = features_nn.numpy().copy()
    

    features_nn_z = scipy.stats.zscore(features_nn.numpy(), axis=0)
    features_nn_z = features_nn_z[:, ~np.isnan(features_nn_z[0,:])]
    features_nn_z = torch.as_tensor(features_nn_z, dtype=torch.float32)
    
    ipcs = -1
    
    for rank in (pbar_pc_sweep := tqdm(pc_sweep)):
        ipcs += 1
        
        fitted_models_dct[jn][rank] = {}
        
        
        pbar_pc_sweep.set_description(f'PCA Rank {rank}')
        
        if rank == 0:
            comp_nn, scores_nn, SVs, EVR_nn = decomposition.torch_pca(features_nn, zscore=False)
        else:
            comp_nn, scores_nn, SVs, EVR_nn = decomposition.torch_pca(features_nn, rank=rank, zscore=False)
        
        features_norm = torch.cat([val / torch.std(val, dim=0).mean() for val in [scores_nn]], dim=1)
        features_train, features_val, labels_train, labels_val = sklearn.model_selection.train_test_split(features_norm, labels_clean, test_size=0.3)

        acc_train, acc_val = [], []
        cm_tr, cm_val, train_cms, test_cms = [], [], [], []
        C_toUse = params_eval['C_toUse']

        splitter = ShuffleSplit(n_splits=classifier_n_splits)
        all_split_inx = list(splitter.split(features_train))

#         train_X = [features_norm[_[0]] for _ in all_split_inx]
#         test_X = [features_norm[_[1]] for _ in all_split_inx]
#         train_y = [labels_clean[_[0]] for _ in all_split_inx]
#         test_y = [labels_clean[_[1]] for _ in all_split_inx]

        train_X = [features_train[_[0]] for _ in all_split_inx]
        test_X = [features_train[_[1]] for _ in all_split_inx]
        train_y = [labels_train[_[0]] for _ in all_split_inx]
        test_y = [labels_train[_[1]] for _ in all_split_inx]
        
        for ic, c in enumerate(C_toUse):
            pbar_pc_sweep.set_description(f'PCA Rank {rank}: {ipcs}/{len(pc_sweep) - 1} — Regularization C {c}')
            
            train_cms_cv = []
            test_cms_cv = []
            
            acc_tr_cv = []
            acc_val_cv = []
            
            for inx_split in range(len(train_X)):
                pbar_pc_sweep.set_description(f'PCA Rank {rank}: {ipcs}/{len(pc_sweep) - 1} — Regularization C {c}: {inx_split}/{len(train_X) - 1}')
                
                tmp_train_X = train_X[inx_split]
                tmp_train_y = train_y[inx_split]

                tmp_test_X = test_X[inx_split]
                tmp_test_y = test_y[inx_split]
                
                logreg = sklearn.linear_model.LogisticRegression(
                    solver='lbfgs',
                    max_iter=logistic_max_iter, 
                    C=c,
                    fit_intercept=True, 
                    class_weight='balanced',
                )
                logreg.fit(tmp_train_X, tmp_train_y)
                

                proba = logreg.predict_proba(tmp_train_X)
                preds = np.argmax(proba, axis=1)
                cm_tr_individual = classification.confusion_matrix(preds, tmp_train_y.astype(np.int32))
                
                acc_tr_cv_individual = logreg.score(tmp_train_X, tmp_train_y.astype(np.int32), sample_weight=get_balanced_sample_weights(tmp_train_y.astype(np.int32)))
                
#                 print('acc_tr_cv_individual', acc_tr_cv_individual)
                acc_tr_cv.append(acc_tr_cv_individual)
                train_cms_cv.append(cm_tr_individual)
                
                
                
                
                proba2 = logreg.predict_proba(tmp_test_X)
                preds2 = np.argmax(proba2, axis=1)
                cm_val2_individual = classification.confusion_matrix(preds2, tmp_test_y.astype(np.int32))
                
                acc_val_cv_individual2 = logreg.score(tmp_test_X, tmp_test_y.astype(np.int32), sample_weight=get_balanced_sample_weights(tmp_test_y.astype(np.int32)))
                
                acc_val_cv.append(acc_val_cv_individual2)
                test_cms_cv.append(cm_val2_individual)
                
                
            acc_train.append(np.array(acc_tr_cv).reshape(1,-1))
            acc_val.append(np.array(acc_val_cv).reshape(1,-1))
            
            train_cms.append(np.expand_dims(np.mean(np.array(train_cms_cv), axis=0), axis=0))
            test_cms.append(np.expand_dims(np.mean(np.array(test_cms_cv), axis=0), axis=0))
            
            
            
            # Refitting model to all of training / CV data and evaluating on heldout data
            logreg_refit = sklearn.linear_model.LogisticRegression(
                solver='lbfgs',
                max_iter=logistic_max_iter, 
                C=c,
                fit_intercept=True, 
                class_weight='balanced',
            )
            logreg_refit.fit(features_train, labels_train)
            logreg_refit_score = logreg_refit.score(features_val, labels_val.astype(np.int32), sample_weight=get_balanced_sample_weights(labels_val.astype(np.int32)))
            fitted_models_dct[jn][rank][c] = {'int': logreg_refit.intercept_,
                                               'coef': logreg_refit.coef_,
                                               'holdout_score': logreg_refit_score}
            
            
            
            
        acc_train = np.concatenate(acc_train, axis=0)
        acc_val = np.concatenate(acc_val, axis=0)
        
        cm_tr = np.concatenate(train_cms,axis=0)
        cm_val = np.concatenate(test_cms,axis=0)


        labels_sesh1 = np.load(path_labels[0])
        labels_sesh2 = np.load(path_labels[1])

        labels_sesh12cat = np.concatenate((labels_sesh1, labels_sesh2), axis=0)
        labels_sesh21cat = np.concatenate((labels_sesh2, labels_sesh1), axis=0)


        layer_1 = model.state_dict()['base_model.0.0.0.0.weight'].cpu()
        

        umap = UMAP(
            n_neighbors=30,
            n_components=2,
            metric='euclidean',
            metric_kwds=None,
            output_metric='euclidean',
            output_metric_kwds=None,
            n_epochs=None,
            learning_rate=1.0,
            init='spectral',
            min_dist=0.1,
            spread=1.0,
            low_memory=True,
            n_jobs=-1,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            random_state=None,
            angular_rp_forest=False,
            target_n_neighbors=-1,
            target_metric='categorical',
            target_metric_kwds=None,
            target_weight=0.5,
            transform_seed=42,
            transform_mode='embedding',
            force_approximation_algorithm=False,
            verbose=False,
            tqdm_kwds=None,
            unique=False,
            densmap=False,
            dens_lambda=2.0,
            dens_frac=0.3,
            dens_var_shift=0.1,
            output_dens=False,
            disconnection_distance=None,
            precomputed_knn=(None, None, None),
        )

        embeddings = umap.fit_transform(features_nn)


        scree_plt_dct[jn][rank] = EVR_nn.detach().numpy()
        corr_plt_dct[jn][rank] = torch.corrcoef(features_nn.T)
        acc_plt_dct[jn][rank] = {'C':C_toUse, 'acc_tr':acc_train, 'acc_val':acc_val}
        cm_plt_dct[jn][rank] = { 'cm_tr':cm_tr,
                           'cm_val':cm_val,
                           'cm_relabel':classification.confusion_matrix(labels_sesh12cat.astype(np.int32), labels_sesh21cat.astype(np.int32))}
        embeddings_plt_dct[jn][rank] = embeddings
        

        features_nn_plt_dct[jn][rank] = {'features_nn': features_nn,
                                   'features_nn_z': features_nn_z,}
        img_grid_plt_dct[jn][rank] = torch.cat([arr for arr in layer_1], dim=0)
        scores_nn_plt_dct[jn][rank] = scores_nn
        labels_clean_plt_dct[jn][rank] = labels_clean


def recursive_list_check_params(dct):
    setup_pp = {}
    for key_a in dct:
        if len(str(key_a) + ": " + str(dct[key_a])) <= 70:
            setup_pp[key_a] = str(dct[key_a])
        elif type(dct[key_a]) is dict:
            setup_pp[key_a] = recursive_list_check_params(dct[key_a])
        else:
            setup_pp[key_a] = dct[key_a]
    return setup_pp

setup_pretty_params = recursive_list_check_params(params)
pretty_params = json.dumps(setup_pretty_params, indent=2)

num_rows = max(len(pc_sweep), 4)

print('scree_plt_dct', scree_plt_dct)

for jn in scree_plt_dct.keys():
    params = params_dct[jn]
    
    fig, ax = plt.subplots(num_rows, 4, figsize=(30,15))
    fig.set_facecolor('white')
    
    meta_data = \
f"""
Params:
Changing — {plot_title_job_num_ids}
{pretty_params}
"""
    
    for tmp_ax_row in range(num_rows):
        ax[tmp_ax_row,0].axis('off')
#     ax[0,0].invert_yaxis()
    ax[num_rows-1,0].annotate(meta_data, xy=(0, 0), size=8)
    
    overall_plot_col = 1
    
    scree_plt = scree_plt_dct[jn][0]
    ax[0,overall_plot_col].set_yscale('log')
    ax[0,overall_plot_col].set_title(f'Scree — {run_id_dct[jn]}')
    ax[0,overall_plot_col].plot(scree_plt)

    ax[1,overall_plot_col].set_title(f'Relabeling — {run_id_dct[jn]}')
    sns.heatmap(
        np.round(cm_plt_dct[jn][0]['cm_relabel'], 3),
        annot=True, 
        annot_kws={"size": 16}, 
        vmax=1., 
        cmap=plt.get_cmap('gray'),
        ax=ax[1,overall_plot_col]
    )

    
    ax[2,overall_plot_col].scatter(embeddings_plt_dct[jn][0][:,0], embeddings_plt_dct[jn][0][:,1],
                s=5, c=labels_clean_plt_dct[jn][0], cmap='gist_rainbow')
    ax[2,overall_plot_col].set_title(f'Relabeling — {run_id_dct[jn]}')
    
    best_mean_acc_val_ovrl = -1.0

    for rank_num in trange(len(pc_sweep)):
        
        best_mean_acc_tr = -1.0
        best_i_C_tr = -1.0
        best_mean_acc_val = -1.0
        best_i_C_val = -1.0

        
        rank = pc_sweep[rank_num]
        print(rank)
        
        title_val = f'{run_id_dct[jn]}, # PCs: {rank}'
        
        acc_ax = ax[rank_num,2]
        cm_ax = ax[rank_num,3]
        
        mean_acc_tr = np.mean(acc_plt_dct[jn][rank]['acc_tr'], axis=1)
        mean_acc_val = np.mean(acc_plt_dct[jn][rank]['acc_val'], axis=1)
        
        for i_C in range(mean_acc_val.shape[0]):
            
            if mean_acc_tr[i_C] > best_mean_acc_tr:
                best_mean_acc_tr = mean_acc_tr[i_C]
                best_i_C_tr = i_C
            if mean_acc_val[i_C] > best_mean_acc_val:
                best_mean_acc_val = mean_acc_val[i_C]
                best_i_C_val = i_C
            
            if best_mean_acc_val > best_mean_acc_val_ovrl:
                best_mean_acc_val_ovrl = np.round(best_mean_acc_val, 3)

        acc_ax.plot(acc_plt_dct[jn][rank]['C'], mean_acc_tr, color='r')
        acc_ax.plot(acc_plt_dct[jn][rank]['C'], mean_acc_val, color='b')
        
        bmacv_plt = np.round(best_mean_acc_val,3)
        
        acc_ax.set_title(f'Acc — {title_val} — Mx Val: {bmacv_plt}')
        acc_ax.set_xscale('log')
        acc_ax.set_xlabel('C')
        acc_ax.set_ylabel('acc')
        acc_ax.legend(['train', 'test']);
        
        for sub_cv in range(acc_plt_dct[jn][rank]['acc_tr'].shape[1]):
            acc_ax.plot(acc_plt_dct[jn][rank]['C'], acc_plt_dct[jn][rank]['acc_tr'][:,sub_cv], color='r', alpha=0.1)
        for sub_cv in range(acc_plt_dct[jn][rank]['acc_val'].shape[1]):
            acc_ax.plot(acc_plt_dct[jn][rank]['C'], acc_plt_dct[jn][rank]['acc_val'][:,sub_cv], color='b', alpha=0.1)
        
        cm_C = acc_plt_dct[jn][rank]['C'][best_i_C_val]
        cm_ax.set_title(f'CM (Val) — {title_val} — C: {cm_C}')
        sns.heatmap(cm_plt_dct[jn][rank]['cm_val'][best_i_C_val], annot=True, annot_kws={"size": 16}, vmax=1., cmap=plt.get_cmap('gray'), ax=cm_ax)

    fig.suptitle(f"{run_id_dct[jn]} — Job #: {jn.split('/')[-1]} — Max Val Acc: {best_mean_acc_val_ovrl}")
    
    
    report_out_name_jn = jn.split('/')[-1]
    fig.savefig(str(Path(dir_save) / f'report_out_jobnum={report_out_name_jn}.png'))
    
    with open(str(Path(dir_save) / f'model_out_jobnum={report_out_name_jn}.pkl'), 'wb') as model_output_file:
        pickle.dump(fitted_models_dct[jn], model_output_file)





