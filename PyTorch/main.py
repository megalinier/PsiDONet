# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 2020

@author: Mathilde Galinier

Pytorch version
"""
import os
import torch
from fundamental_functions.auxiliary_functions import create_path_save_name
from fundamental_functions.Train_Test_PsiDONet import PsiDONet_class
from fundamental_functions.tools import compute_quality_results

# Configuration
missing_angle      = 40
step_angle         = 1
size_image         = 128
mu                 = 0.000002
L                  = 5
train_conditions   = [missing_angle, step_angle, size_image, mu, L]

# Hyperparameters
model_unrolling    ='PSIDONetO'       # 'PSIDONetO' or 'PSIDONetOplus'
learning_rate      = 0.0001
nb_epochs          = 1
minibatch_size     = 5
loss_type          ='MSE'              # 'MSE' or 'SSIM' 
loss_domain        ='WAV'              # 'WAV' or 'IM'   
nb_unrolledBlocks  = 2
nb_repetBlock      = 2
filter_size        = 11                # must be odd
wavelet_type       ='haar'             # 'haar' or 'db2'
level_decomp       = 3
precision_float    = 32
size_val_limit     = 2*minibatch_size
dataset            = 'Apples'

# Path to training set and save folder
optionalText       = 'temp'
path_main          = os.path.join('D:','Helsinki_project')
path_save          = os.path.join(path_main,'Results',\
                     create_path_save_name(train_conditions, optionalText, model_unrolling,
                            learning_rate, nb_epochs, minibatch_size, loss_type, loss_domain,
                            nb_unrolledBlocks, nb_repetBlock, filter_size,
                            wavelet_type, level_decomp, precision_float, dataset))
path_datasets      = os.path.join(path_main,dataset + '_Datasets','Size_'+str(size_image))
paths              = [path_main, path_datasets, path_save]

#%% Training
network           = PsiDONet_class(\
                    train_conditions=train_conditions,\
                    folders=paths,\
                    mode='train',\
                    model_unrolling=model_unrolling,\
                    learning_rate=learning_rate,\
                    nb_epochs=nb_epochs,\
                    minibatch_size=minibatch_size,\
                    loss_type=loss_type,\
                    loss_domain=loss_domain,\
                    nb_unrolledBlocks=nb_unrolledBlocks,\
                    nb_repetBlock=nb_repetBlock,\
                    filter_size=filter_size,\
                    wavelet_type=wavelet_type,\
                    level_decomp=level_decomp,\
                    precision_float=precision_float,\
                    size_val_limit=size_val_limit,\
                    dataset=dataset) 
network.train()

#%% Test
## Load a model
network          = PsiDONet_class(
                    train_conditions=train_conditions,\
                    folders=paths,\
                    mode='test',\
                    model_unrolling=model_unrolling,\
                    learning_rate=learning_rate,\
                    nb_epochs=nb_epochs,\
                    minibatch_size=minibatch_size,\
                    loss_type=loss_type,\
                    loss_domain=loss_domain,\
                    nb_unrolledBlocks=nb_unrolledBlocks,\
                    nb_repetBlock=nb_repetBlock,\
                    filter_size=filter_size,\
                    wavelet_type=wavelet_type,\
                    level_decomp=level_decomp,\
                    precision_float=precision_float,\
                    size_val_limit=size_val_limit,\
                    dataset=dataset)
    
# path_model       = os.path.join(path_save,'models','trained_model_MinLossOnVal.pt')
path_model       = os.path.join(path_save,'models','MinOnVal.pt')
network.model.load_state_dict(torch.load(path_model))
print('--------------------------------------------------------------------------------------------------------------------------------')
print('Loaded model from ' + str(path_model) + '.')
print('--------------------------------------------------------------------------------------------------------------------------------')

network.test()

#%%
## Evaluate the results
print('--------------------------------------------------------------------------------------------------------------------------------')
print('Evaluating the results on test set...')
relative_err_mean, MSE_mean, SSIM_mean, PSNR_mean, HaarPSI_mean \
    = compute_quality_results(os.path.join(path_datasets, 'test','Images'),os.path.join(path_save,'testset_restoredImages'),precision_float)
print('--------------------------------------------------------------------------------------------------------------------------------')
