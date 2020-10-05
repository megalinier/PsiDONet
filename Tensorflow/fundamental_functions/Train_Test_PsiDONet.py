# -*- coding: utf-8 -*-
"""
PSIDONET. Train and test.
Tensorflow version.

Functions
-----------
        train                   :  trains a PsiDONet model
        test                    :  uploads and tests a trained PsiDONet model                       
                            
@author: Mathilde Galinier
@date: 29/09/2020
"""
import os
import sys
import numpy as np
import tensorflow as tf
from auxiliary_functions import createFolders, compute_size_folder, save_parameters
from tools               import compute_angles, get_nbDetectorPixels, dtypes, \
                                LoadData
from utils_bowtie        import initialise_parameters, restore_parameters
import tf_wavelets       as tfw
import utils_learning
import pywt
import scipy.io as sio

def train(train_conditions, folders, model_unrolling ='PSIDONetO',
        lr= 0.001, nb_epochs=3, minibatch_size=15, loss_type='MSE', loss_domain = 'WAV',
        nb_unrolledBlocks=40, nb_repetBlock=3, filter_size = 64,
        wavelet_type = 'haar', level_decomp = 3 , precision_float = 32, size_val_limit = 60,
        dataset = 'Ellipses', path_to_restore = '', start_from_epoch=0):
        """
        Trains a PsiDONet model.
        Parameters
        -----------
            train_conditions (list)   : list containing [missing_angle, step_angle, size_image, mu, L] where:
                                            missing_angle (int)  : missing angle on the first quadrant
                                            step_angle  (int)    : number of degrees between two consecutive 
                                                                   projections (>=1). 
                                            size_image (int)     : image dimension (multiple of 2)
                                            mu (float)           : standard ISTA regularisation parameter
                                            L (float)            : standard ISTA constant
            folders (list)            : list containing [path_main, path_datasets, path_save] where
                                            path_main (string)   : path to main folder containing all the 
                                                                   subfolders to be used
                                            path_datasets(string): path to the dataset
                                            path_save (string)   : path where to save the results
            model_unrolling (string)  : name of the version of PsiDONet used to train/test the model
                                        ('PSIDONetO', 'PSIDONetOplus', 'PSIDONetF' or 'PSIDONetFplus')
            learning_rate (float)     : learning rate used for the training
            nb_epochs (int)           : number of epochs used for the training
            minibatch_size (int)      : size of the minibatch used for the training
            loss_type (string)        : kind of cost function 
            loss_domain (string)      : domain in which the evaluation of the cost function takes place
            nb_unrolledBlocks (int)   : number of different set of trainable parameters 
            nb_repetBlock (int)       : number of times each set of trainable parameters is used 
            filter_size (int)         : size of each trainable filter
            wavelet_type (string)     : type of wavelets
            level_decomp (int)        : number of wavelet scales (J-J0-1 in the article)
            precision_float (int)     : machine precision (16 for half-precision, 
                                        32 for single-precision, 64 for double-precision)
            size_val_limit(int)       : size of the subset taken from the validation set, used to evaluate 
                                        the performance of the trained model during its training (for early stopping)
            dataset (string)          : dataset of interest 
            path_to_restore (string)  : [optional] path to previously trained model, with the aim to 
                                        retore it and keep training it. If not model is to be restored,
                                        path_to_restore is set to ''.
            start_from_epoch (int)    : [optional] if a model is restored, indicates at which epoch its 
                                        training was left. 0 otherwise.
        Returns
        -----------        
           (float)                    : name of the folder where to save the results 
        """
        print('=================== Training model ===================')
        
        # Additional hyperpameter
        missing_angle, step_angle, size_image, mu, L    = train_conditions
        path_main, path_datasets, path_save             = folders
        angles            = compute_angles(missing_angle, step_angle)
        path_train        = os.path.join(path_datasets,'train')
        path_val          = os.path.join(path_datasets,'val')
        nbDetectorPixels  = get_nbDetectorPixels(path_datasets)
        modulo_batch      = 25 #the stats of the training process are computed each modulo_batch minibtaches
        size_train        = compute_size_folder(os.path.join(path_train,'Images'))
        mode              = 'train'
        tf_prec, np_prec  =  dtypes(precision_float)
        
        if path_to_restore=='':
            costs = []   
            costs_val = []  
        else:
            costs = list(np.load(path_to_restore + '/costs.npy'))
            costs_val = list(np.load(path_to_restore + '/costs_val.npy'))
      
        # Compute rescaling coefficients
        rescal_scikit     = utils_learning.Get_Rescal_Factor(size_image,angles,'scikit', level_decomp)

        if 'F' in model_unrolling: # Filter-based implementation. 
            rescal = utils_learning.Get_Rescal_Factor(size_image,angles, 'bowtie', level_decomp, np_prec, \
                                                       tf_prec, precision_float, nbDetectorPixels, wavelet_type)
        elif 'O' in model_unrolling: # Operator-based implementation
            rescal = utils_learning.Get_Rescal_Factor(size_image,angles, 'astra', level_decomp, np_prec, tf_prec,\
                                                      precision_float, nbDetectorPixels)
                   
        # Create Folder to save results
        createFolders(mode,path_save)
  
        # To find the minimum cost model
        cost_min_val      =  float('Inf')                   
        
        #Create a new graph and fix seeds
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(1) 
        np.random.seed(3)
        
        # Compute wavelet filters
        wavelet_filter = tfw.create_wavelet_transform_filters(wavelet_type, tf_prec, np_prec) 
        
        # Create the placeholders
        wave_bp_plc   = tf.compat.v1.placeholder(tf_prec, shape=(minibatch_size, size_image, size_image, 1), name = 'wave_bp')
        wave_true_plc = tf.compat.v1.placeholder(tf_prec, shape=(minibatch_size, size_image, size_image, 1), name = 'wave_true')
        lr_plc        = tf.compat.v1.placeholder(tf_prec, shape=[], name = 'lr')
        
        # Loading the small (fixed) validation set 
        indices_small_val = np.linspace(0, size_val_limit-1, size_val_limit).astype(int)
        _, wave_bp_val , wave_true_val = LoadData(angles, indices_small_val, size_image, \
                                                           level_decomp, 'wave', wavelet_type,\
                                                           rescal_scikit, path_val, np_prec)

         # Initialise or restore variables
        if path_to_restore == '': #initialisation (nothing to restore)
            dictionaries_tf, thetas_tf, alphas_tf, betas_tf = initialise_parameters(filter_size, mu, L, \
                                                                                    nb_unrolledBlocks, model_unrolling,\
                                                                                    size_image, angles, level_decomp, \
                                                                                    tf_prec, np_prec, wavelet_type)
        else: #restoration of existing model
            dictionaries_tf, thetas_tf, alphas_tf, betas_tf = restore_parameters(path_to_restore, nb_unrolledBlocks,\
                                                                                 tf_prec, np_prec, 1)
        
        # Forward propagation: Build the forward propagation in the tensorflow graph
        wave_pred_plc  = getattr(utils_learning,model_unrolling)\
                        (wave_bp_plc, size_image, level_decomp, angles, dictionaries_tf, thetas_tf, \
                        alphas_tf, betas_tf, filter_size, nb_unrolledBlocks, nb_repetBlock, wavelet_type, \
                        wavelet_filter, nbDetectorPixels, rescal, tf_prec, np_prec, precision_float)

        # Cost function: Add cost function to tensorflow graph
        if loss_domain == 'WAV':
            my_prediction = wave_pred_plc
            my_true = wave_true_plc
        elif loss_domain == 'IM':
            #reshape and permutations in order to apply tfw
            wave_pred_perm = tf.transpose(tf.squeeze(wave_pred_plc,[3]), perm = [1,2,0])
            wave_true_perm = tf.transpose(tf.squeeze(wave_true_plc,[3]), perm = [1,2,0])
            #Going back to the image domain
            im_pred = tfw.idwt2d(wave_pred_perm, wavelet=wavelet_filter, levels=level_decomp)
            im_true = tfw.idwt2d(wave_true_perm, wavelet=wavelet_filter, levels=level_decomp)
            # Permutations to have objects with size [batch, size, size, 1]
            my_prediction = tf.transpose(im_pred, perm=[2,0,1])[...,np.newaxis]
            my_true = tf.transpose(im_true, perm=[2,0,1])[...,np.newaxis]
         
        cost = utils_learning.compute_cost(my_prediction, my_true, loss_type)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        opt_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_plc) #GradientDescentOptimizer
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = opt_adam.minimize(cost)
            
        # Initialize all the variables globally
        init = tf.compat.v1.global_variables_initializer()
        options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
    
        # Start the session to compute the tensorflow graph
        with tf.compat.v1.Session() as sess:
            
            # Run the initialization
            print('Initialisation of the variables...')
            sess.run(init)
            
            # Save the initial parameters
            dictionaries, thetas, alphas, betas = sess.run([dictionaries_tf, thetas_tf, alphas_tf, betas_tf])
            save_parameters(os.path.join(path_save,'parameters', \
                                         'parameters_init'),\
                                         dictionaries, thetas, alphas, betas)

            # Do the training loop
            print('Starting minimisation process...') 
            for epoch in range(nb_epochs):
                permutation = list(np.random.permutation(size_train))
                shuffled_ind_samples = np.array(np.linspace(0, size_train-1, size_train).astype(int))[permutation]
                
                if epoch>0:
                    lr = lr*0.9
                
                nb_minibatches = int(np.ceil(size_train/minibatch_size))
                for num_minibatch in range(nb_minibatches):
                    ind_minibatch =  shuffled_ind_samples[num_minibatch*minibatch_size:\
                                                         (num_minibatch+1)*minibatch_size]
                    
                    # Loads a training minibatch
                    names, wave_bp_minibatch , wave_true_minibatch = LoadData(angles, ind_minibatch, size_image, \
                                                                       level_decomp, 'wave', wavelet_type,\
                                                                       rescal_scikit, path_train, np_prec)
                    if len(names)<minibatch_size:
                        break

                    # Runs the optimisation 
                    _ , temp_cost = sess.run([optimizer,cost],\
                                    feed_dict = {wave_bp_plc:wave_bp_minibatch, \
                                                 wave_true_plc:wave_true_minibatch,\
                                                 lr_plc: lr},\
                                                 options=options, run_metadata=run_metadata)
                    dictionaries, thetas, alphas, betas = sess.run([dictionaries_tf, thetas_tf, alphas_tf, betas_tf])
                    costs.append(temp_cost)     
                    np.save(path_save+'/training_stats/costs',costs)
                    
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,num_minibatch,temp_cost))
                    
                    # Computation of the cost on the small validation set
                    if num_minibatch%modulo_batch==0: 
                        cost_current_val = 0.
                        
                        #Dividing in minibatches
                        nb_minibatches_val = int(np.ceil(size_val_limit/minibatch_size))
                        for num_minibatch_val in range(nb_minibatches_val):
                            ind_minibatch_val =  (np.linspace(num_minibatch_val*minibatch_size,
                                                 (num_minibatch_val+1)*minibatch_size-1,minibatch_size)).astype(int)   
                            # Computation of the cost
                            temp_cost_val       = sess.run(cost, \
                                                           feed_dict = {wave_bp_plc : wave_bp_val[ind_minibatch_val,:,:,:],\
                                                                      wave_true_plc:wave_true_val[ind_minibatch_val,:,:,:]})
                            cost_current_val+= temp_cost_val                                                                
                        sys.stdout.write('\r(%d) validation loss: %5.4f '%(epoch,cost_current_val))
                            
                        cost_current_val = cost_current_val/nb_minibatches_val  
                        costs_val.append(cost_current_val)    
                        np.save(path_save+'/training_stats/costs_val',costs_val)    
                     
                        # Saving the model minimising the cost function
                        if cost_min_val>cost_current_val:
                            save_parameters(os.path.join(path_save,'parameters','MinOnVal'), dictionaries, thetas, alphas, betas)
                            cost_min_val = cost_current_val    
                                  
                # save costs and parameters and model
                save_parameters(os.path.join(path_save,'parameters', \
                                             'parameters_ep'+str(epoch+start_from_epoch)),\
                                             dictionaries, thetas, alphas, betas)
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(path_save,'models', 'graph_ep'+str(epoch+start_from_epoch)))
    
        # training is finished
        print('-----------------------------------------------------------------')
        print('Training is done.')
        print('-----------------------------------------------------------------')


def test(train_conditions, folders, model_unrolling ='PSIDONetO', minibatch_size=15,
        nb_unrolledBlocks=40, nb_repetBlock=3, filter_size = 64, 
        wavelet_type = 'haar', level_decomp = 3 , precision_float = 32, 
        dataset = 'Ellipses', path_to_restore = ''):
        """
        Tests a PsiDONet model.
        Parameters
        -----------
            train_conditions (list)   : list containing [missing_angle, step_angle, size_image, mu, L] where:
                                            missing_angle (int)  : missing angle on the first quadrant
                                            step_angle  (int)    : number of degrees between two consecutive 
                                                                   projections (>=1). 
                                            size_image (int)     : image dimension 
                                            mu (float)           : standard ISTA regularisation parameter
                                            L (float)            : standard ISTA constant
            folders (list)            : list containing [path_main, path_datasets, path_save] where
                                            path_main (string)   : path to main folder containing all the 
                                                                   subfolders to be used
                                            path_datasets(string): path to the dataset
                                            path_save (string)   : path where to save the results
            model_unrolling (string)  : name of the version of PsiDONet used to train/test the model
                                        ('PSIDONetO', 'PSIDONetOplus', 'PSIDONetF' or 'PSIDONetFplus')
            nb_unrolledBlocks (int)   : number of different set of trainable parameters 
            nb_repetBlock (int)       : number of times each set of trainable parameters is used 
            filter_size (int)         : size of each trainable filter
            wavelet_type (string)     : type of wavelets
            level_decomp (int)        : number of wavelet scales (J-J0-1 in the article)
            precision_float (int)     : machine precision (16 for half-precision, 
                                        32 for single-precision, 64 for double-precision)
            dataset (string)          : dataset of interest 
            path_to_restore (string)  : path to model to be tested
        Returns
        -----------        
           (float)                    : name of the folder where to save the results 
        """
        print('=================== Testing model ===================')
        
        # Additional hyperpameter
        missing_angle, step_angle, size_image, mu, L    = train_conditions
        path_main, path_datasets, path_save             = folders
        angles            = compute_angles(missing_angle, step_angle)
        mode              = 'test'
        tf_prec, np_prec  =  dtypes(precision_float)
        nbDetectorPixels  = get_nbDetectorPixels(path_datasets)

        # Compute rescaling coefficients
        rescal_scikit     = utils_learning.Get_Rescal_Factor(size_image,angles,'scikit', level_decomp)
        
        if 'F' in model_unrolling: # Filter-based implementation. 
            rescal = utils_learning.Get_Rescal_Factor(size_image,angles, 'bowtie', level_decomp, np_prec, \
                                                    tf_prec, precision_float, nbDetectorPixels, wavelet_type)
        elif 'O' in model_unrolling: # Operator-based implementation
            rescal = utils_learning.Get_Rescal_Factor(size_image,angles,'astra',level_decomp, np_prec, tf_prec, \
                                                    precision_float, nbDetectorPixels)
        
        # Create Folder to save results
        createFolders(mode,path_save)

        #Create a new graph and fix seeds
        tf.compat.v1.reset_default_graph()

        # Compute wavelet filters
        wavelet_filter = tfw.create_wavelet_transform_filters(wavelet_type, tf_prec, np_prec) 
        
        # Create the placeholders
        wave_bp_plc   = tf.compat.v1.placeholder(tf_prec, shape=(minibatch_size, size_image, size_image, 1), name = 'wave_bp')
        wave_true_plc = tf.compat.v1.placeholder(tf_prec, shape=(minibatch_size, size_image, size_image, 1), name = 'wave_true')

        # Restoration of the parameters
        dictionaries_tf, thetas_tf, alphas_tf, betas_tf = restore_parameters(path_to_restore, nb_unrolledBlocks,\
                                                        tf_prec, np_prec, 0)
                
        # Forward propagation: Build the forward propagation in the tensorflow graph
        wave_pred_plc  = getattr(utils_learning,model_unrolling)\
                        (wave_bp_plc, size_image, level_decomp, angles, dictionaries_tf, thetas_tf, \
                        alphas_tf, betas_tf, filter_size, nb_unrolledBlocks, nb_repetBlock, wavelet_type, \
                        wavelet_filter, nbDetectorPixels, rescal, tf_prec, np_prec, precision_float)          
                        
        # Initialize all the variables globally
        init = tf.compat.v1.global_variables_initializer()
    
        # Start the session to compute the tensorflow graph
        with tf.compat.v1.Session() as sess:
            # Run the initialization
            sess.run(init)
            
            for folder_to_test in ['test','val']:
                path_test         = os.path.join(path_datasets,folder_to_test)
                size_test         = compute_size_folder(os.path.join(path_test,'Images'))

                nb_minibatches = int(np.ceil(size_test/minibatch_size))
                for num_minibatch in range(nb_minibatches):
                    ind_minibatch =  (np.linspace(num_minibatch*minibatch_size,
                                                        (num_minibatch+1)*minibatch_size-1,minibatch_size)).astype(int)   
                    names, wave_bp, wave_true= LoadData(angles, ind_minibatch, size_image, \
                                                        level_decomp, 'wave', wavelet_type,\
                                                        rescal_scikit, path_test, np_prec)
                    
                    if len(names)<minibatch_size:
                        break

                    # Compute predictions in the wavelet domain
                    wave_pred = sess.run(wave_pred_plc,\
                                        feed_dict = {wave_bp_plc   : wave_bp, \
                                                    wave_true_plc : wave_true})   
                    print('Numero of the tested minibatch: ' + str(num_minibatch+1) + '/' + str(nb_minibatches))                                 
                    
                    # save images
                    for j in range(minibatch_size):
                        _, coeff_slices = pywt.coeffs_to_array(pywt.wavedecn(np.zeros((size_image,size_image)), wavelet_type, \
                                                            mode='periodization', level=level_decomp))
                        coeffs_from_arr = pywt.array_to_coeffs(wave_pred[j,:,:,0],coeff_slices)
                        im_pred = pywt.waverecn(coeffs_from_arr, wavelet=wavelet_type, mode = 'periodization')
                        
                        sio.savemat(os.path.join(path_save,folder_to_test+'set_restoredImages',names[j]),{'image':im_pred})                   
