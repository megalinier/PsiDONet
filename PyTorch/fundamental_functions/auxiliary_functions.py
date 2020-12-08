# -*- coding: utf-8 -*-
"""
Auxiliary functions.
PyTorch version.

Functions
-----------
        convert_float_to_string     :    converts a float into a string, and removes the point if 
                                         float is <1
        create_path_save_name       :    generates the name (string) of the folder where the results 
                                         are to be saved
                            
@author: Mathilde Galinier
@date: 07/12/2020
"""

def convert_float_to_string(x):
    """
    Converts a float into a string. If x is <1, the point is removed and the 
    float is truncated as soon as the first non-zero decimal is encountered.
    Parameters
    -----------
        x (float)       : float to be converted
    Returns
    -----------        
       my_string(float) : string 
    """    
    if isinstance(x, str):
        return x
    if x>=1:
        my_string = str(int(x))
    elif x==0:
        my_string = '0'
    else:
        y = x
        my_string = ''
        while(y<1):
            y = y*10
            my_string = '0' + my_string 
        my_string = my_string + str(int(y))    
    return my_string           

def create_path_save_name(train_conditions, optionalText, model_unrolling ='PSIDONet-O',
                learning_rate= 0.001, nb_epochs=3, minibatch_size=15, loss_type='MSE', 
                loss_domain = 'WAV', nb_unrolledBlocks=40, nb_repetBlock=3, filter_size = 64, 
                wavelet_type = 'haar', level_decomp = 3 , precision_float = 32, 
                dataset = 'Ellipses'):
    """
    Generates the name of the folder where the results are to be saved.
    Parameters
    -----------
        train_conditions (list)   : list containing [missing_angle, step_angle, size_image, mu, L]
        optionalText (string)     : additional text that can be added at the end of the generated string
        model_unrolling (string)  : name of the version of PsiDONet used to train/test the model
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
        dataset (string)          : dataset of interest 
    Returns
    -----------        
       (string)                   : name of the folder where to save the results 
    """    
    return 'Angl_' + str(train_conditions[0]) + '_' + str(180-train_conditions[0])\
        +'_step' + str(train_conditions[1]) + '_sizeIm' + str(train_conditions[2])\
        + '_waveType' + wavelet_type\
        + '_levelDec' + str(level_decomp)\
        + '_lr' +  str(convert_float_to_string(learning_rate))\
        + '_nepochs' + str(nb_epochs) \
        + '_sizebatch' + str(minibatch_size)  \
        + '_nbUnrolledBlocks' + str(nb_unrolledBlocks)\
        + '_nbrepetBlock' + str(nb_repetBlock)\
        + '_mu' + convert_float_to_string(train_conditions[3])\
        + '_L' + convert_float_to_string(train_conditions[4])\
        + '_' + loss_type + loss_domain\
        + '_filterSize' + str(filter_size)\
        + '_' + dataset\
        + '_' + str(model_unrolling)\
        + optionalText  
                
                