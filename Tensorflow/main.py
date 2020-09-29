# -*- coding: utf-8 -*-
"""
Created in Sept 2020

@author: Mathilde Galinier

Tensorflow version
"""
import sys
sys.path.append('/work/galinier_tomography_project/Helsinki_project/Tensorflow/fundamental_functions') 

import os
from auxiliary_functions import create_path_save_name
from Train_Test_PsiDONet import train, test
from tools import compute_quality_results

# Configuration
missing_angle      = 30
step_angle         = 1
size_image         = 128
mu                 = 0.000002
L                  = 5
train_conditions   = [missing_angle, step_angle, size_image, mu, L]

print('Launch of the inter-galactic algorithm...')

# Hyperparameters
for model_unrolling  in ['PSIDONetF','PSIDONetO','PSIDONetFplus','PSIDONetOplus']:        # 'PSIDONetF' or 'PSIDONetFplus' or 'PSIDONetO' or 'PSIDONetOplus'
        for learning_rate  in [0.005]:
                nb_epochs          = 3
                minibatch_size     = 25
                loss_type          ='MSE'              # 'MSE' or 'SSIM' 
                loss_domain        ='WAV'              # 'WAV' or 'IM'   
                nb_unrolledBlocks  = 40
                nb_repetBlock      = 3
                filter_size        = size_image//4            # 'astra' 
                wavelet_type       ='haar'             # 'haar' or 'db2'
                level_decomp       = 3
                precision_float    = 32
                size_val_limit     = 4*minibatch_size
                for dataset  in  ['Ellipses', 'AppleCT']:            

                        # Path to training set and save folder
                        optionalText       = ''
                        path_main          = os.path.join('/work/galinier_tomography_project/Helsinki_project')
                        path_save          = os.path.join(path_main,'Tensorflow','Results',\
                                        create_path_save_name(train_conditions, optionalText, model_unrolling,
                                                learning_rate, nb_epochs, minibatch_size, loss_type, loss_domain,
                                                nb_unrolledBlocks, nb_repetBlock, filter_size, 
                                                wavelet_type, level_decomp, precision_float, dataset))
                        path_datasets      = os.path.join(path_main,dataset + '_Datasets','Size_'+str(size_image))
                        paths              = [path_main, path_datasets, path_save]

                        # #%% Training
                        train(train_conditions=train_conditions,\
                                folders=paths,\
                                model_unrolling=model_unrolling,\
                                lr=learning_rate,\
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
                                dataset=dataset, \
                                path_to_restore = '',\
                                start_from_epoch = 0)

                        #%% Test
                        #Model to Load
                        path_to_restore = os.path.join(path_save,'parameters','MinOnVal')

                        test(train_conditions=train_conditions,\
                                folders=paths,\
                                model_unrolling=model_unrolling,\
                                minibatch_size=minibatch_size,\
                                nb_unrolledBlocks=nb_unrolledBlocks,\
                                nb_repetBlock=nb_repetBlock,\
                                filter_size=filter_size,\
                                wavelet_type=wavelet_type,\
                                level_decomp=level_decomp,\
                                precision_float=precision_float,\
                                dataset=dataset, \
                                path_to_restore = path_to_restore)  

                        #%%
                        ## Evaluate the results
                        print('--------------------------------------------------------------------------------------------------------------------------------')
                        print('Evaluating the results on test set...')
                        relative_err_mean, MSE_mean, SSIM_mean, PSNR_mean, HaarPSI_mean \
                        = compute_quality_results(os.path.join(path_datasets, 'test','Images'),os.path.join(path_save,'testset_restoredImages'),precision_float)
                        print('--------------------------------------------------------------------------------------------------------------------------------')

                        print('--------------------------------------------------------------------------------------------------------------------------------')
                        print('Evaluating the results on test set...')
                        relative_err_mean, MSE_mean, SSIM_mean, PSNR_mean, HaarPSI_mean \
                        = compute_quality_results(os.path.join(path_datasets, 'val','Images'),os.path.join(path_save,'valset_restoredImages'),precision_float)
                        print('--------------------------------------------------------------------------------------------------------------------------------')
