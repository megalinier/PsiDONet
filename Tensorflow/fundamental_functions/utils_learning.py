# -*- coding: utf-8 -*-
"""
Functions for learning process.
Tensorflow version.

Functions
-----------
        unpool            :  zeros-unpooling operation
        ConvFilters       :  applies the downsampling, upsampling and convolutions as shown on fig. 6 of article
        compute_cost      :  computes the loss function
        WRstarRWstar      :  computes the operator WR*RW* using the astra-toolbox
        Get_Rescal_Factor :  computes the rescaling factor, that is the inverse of the norm of 
                             the normal operator of the radon transform
        PSIDONetF         :  builds the graph of PSIDONet-F
        PSIDONetFplus     :  builds the graph of PSIDONet-F+
        PSIDONetO         :  builds the graph of PSIDONet-O
        PSIDONetOplus     :  builds the graph of PSIDONet-O+
@author: Mathilde Galinier
@date: 29/09/2020
"""

import numpy as np
import tensorflow as tf
import tf_wavelets as tfw
import odl
import odl.contrib.tensorflow
import math
from skimage.transform import iradon, radon
from utils_bowtie import initialize_filters_F, creation_dict_bowties,\
                         separate_dict_hole_NN

def unpool(value, factor, name='unpool'):
    """
    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    Parameters
    -----------
      value (tensor)        : tensor to be unpooled, shape b*h*w*c
      factor (int)          : power of 2 by which the size of value has to be increased
      name (string)         : [optional]   
    Returns
    -----------        
       out (tensor)         : unpooled tensor, shape b*2h*2w*c
    """  
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for f in range(factor):
            for i in range(dim, 0, -1):
                out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * (2**factor) for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

    
def ConvFilters(w, size_image, level_decomp, dict_filters, rescal, tf_prec):
    """
    Applies the downsampling, upsampling and convolution operations as shown on fig. 6 of article
    Parameters
    -----------
        w (tensor)                : wavelet object to which apply the operations, shape b*h*w*1
        size_image (int)          : object dimension (multiple of 2)
        level_decomp (int)        : number of wavelet scales (J-J0-1 in the article)           
        dict_filters (dict)       : dictionary containing the convolutional filters (tensorflow variables)
        rescal (int)              : rescaling factor so that the normal operator of the radon transform has unit norm
        tf_prec(tensorflow.dtype) : tensorflow type corresponding to the desired machine precision      
    Returns
    -----------        
        (tensor)                  : wavelet object, shape b*h*w*1
    """    
    #initialize the output
    w_next = tf.zeros(tf.shape(w), dtype=tf_prec)

    if level_decomp==0:
       filter_tensor = dict_filters['F_0_0_0']
       w_next =  tf.nn.conv2d(w, filter_tensor, \
                                   strides = [1,1,1,1], padding = 'SAME') 
    else:
        # In the loop: each block is convolved with the bunch of corresponding filters
        for level_block in range(1,level_decomp+1):
            size_block = int(size_image/(2**level_block))
            for position_block in ['v','h','d','l']:   
                if level_block<level_decomp and position_block=='l':
                    continue
                else:
                    if position_block == 'h':
                        indices_pos_x_0 = 0
                        indices_pos_y_0 = size_block
                    elif position_block == 'v':    
                        indices_pos_x_0 = size_block  
                        indices_pos_y_0 = 0 
                    elif position_block == 'd':    
                        indices_pos_x_0 = size_block 
                        indices_pos_y_0 = size_block  
                    elif position_block == 'l':    
                        indices_pos_x_0 = 0
                        indices_pos_y_0 = 0
                
                    block_init = w[:,indices_pos_x_0:indices_pos_x_0+size_block,\
                                indices_pos_y_0:indices_pos_y_0+size_block,:]             
                    
                    for level_filter in range(1,level_decomp+1):
                        size_block_result = int(size_image/(2**level_filter))
                        adapted_block_computed = False
                        for position_filter in ['d','h','v','l']:   
                            if level_filter<level_decomp and position_filter=='l':
                                continue
                            else:             
                                ratio_levels = level_block - level_filter
                                if ratio_levels>0 and not(adapted_block_computed): #--> upsampling
                                    block_adapted = unpool(block_init, ratio_levels)
                                    adapted_block_computed = True
                                elif ratio_levels<0 and not(adapted_block_computed): #--> downsampling
                                    fact = 2**(-ratio_levels) 
                                    block_adapted = tf.nn.avg_pool(block_init,\
                                                        [1,fact,fact,1],[1,fact,fact,1],padding='VALID')
                                    adapted_block_computed = True
                                elif ratio_levels==0: #--> we keep the same size
                                    block_adapted = block_init
                                    
                                name_filter = 'F_' + str(level_block) +'_'+ position_block +'_'+ str(level_filter) +'_'+ position_filter
                                if position_filter == 'h':
                                    paddings = tf.constant([[0,0],[0,size_image-size_block_result,], \
                                                [size_block_result, size_image-2*size_block_result],[0,0]])
                                elif position_filter == 'v':    
                                    paddings = tf.constant([[0,0],[size_block_result, size_image-2*size_block_result],\
                                                [0,size_image-size_block_result,], [0,0]])
                                elif position_filter == 'd':    
                                    paddings = tf.constant([[0,0],[size_block_result, size_image-2*size_block_result], \
                                                [size_block_result, size_image-2*size_block_result],[0,0]])
                                elif position_filter == 'l':    
                                    paddings = tf.constant([[0,0],[0,size_image-size_block_result], \
                                                            [0,size_image-size_block_result],[0,0]])
                                
                                # Convolution with the central part of the filter 
                                filter_tensor = dict_filters[name_filter]
                                block_result =  tf.nn.conv2d(block_adapted, filter_tensor, \
                                                            strides = [1,1,1,1], padding = 'SAME')

                                block_result_padded = tf.pad(block_result, paddings, mode='CONSTANT')
                                w_next= tf.add(w_next,block_result_padded)                            
    return w_next * rescal

def compute_cost(my_pred, my_true, loss_type):
    """
    Computes the cost function
    Parameters
    -----------
        my_pred (tensor)    : predicted object, shape b*h*w*1
        my_true (tensor)    : ground truth object, shape b*h*w*1
        loss_type (string)  : 'MSE', 'L1' or 'SSIM'
    Returns
    -----------        
       cost(tensor)         : cost function
    """   
    if loss_type == 'MSE':
        cost = tf.compat.v1.losses.mean_squared_error(my_true,my_pred,weights=1.0)
    elif loss_type == 'L1':
        cost = tf.compat.v1.losses.absolute_difference(my_true,my_pred,weights=1.0)
    elif loss_type == 'SSIM':
        cost = 1 - tf.compat.v1.reduce_sum(tf.compat.v1.image.ssim(my_true,my_pred,.0))
    return cost

   
def WRstarRWstar(w, size_image, wavelet_filter, level_decomp, angles, nbDetectorPixels,
                        rescal, tf_prec, precision_float):
    """
    Applies the operator WR*RW* (astra-toolbox)
    Parameters
    -----------
        w (tensor)                 : wavelet object to which apply the operations, shape b*h*w*1
        size_image (int)           : object dimension (multiple of 2)
        wavelet_filter(tfw.Wavelet): wavelet coefficients for the computation of the wavelet transform
        level_decomp (int)         : number of wavelet scales (J-J0-1 in the article) 
        angles(numpy array)        : angles of the 'seen' projections          
        nbDetectorPixels (int)     : sinogram width 
        rescal (int)               : rescaling factor so that the normal operator of the radon transform has unit norm
        tf_prec(tensorflow.dtype)  : tensorflow type corresponding to the desired machine precision
        precision_float (int)      : machine precision (16 for half-precision, 
                                    32 for single-precision, 64 for double-precision)
    Returns
    -----------        
        (tensor)                  : wavelet object, shape b*h*w*1
    """    
    precision_float_str = 'float' + str(precision_float)

    space = odl.uniform_discr([-size_image//2, -size_image//2], [size_image//2, size_image//2], [size_image, size_image],
                          dtype=precision_float_str)
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(math.radians(angles[0]+90), math.radians(angles[-1]+90), angles.shape[0])
    
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(-nbDetectorPixels//2, nbDetectorPixels//2, nbDetectorPixels)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    ray_transform = odl.tomo.RayTransform(space, geometry, impl = 'astra_cuda')
  
    # Create tensorflow layer from odl operator
    odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(ray_transform, 'RayTransform')
    odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(ray_transform.adjoint, 'RayTransformAdjoint')
    w_perm = tf.transpose(tf.squeeze(w,[3]), perm = [1,2,0])

    # Go to image domain
    Wstar_w = tfw.idwt2d(w_perm, wavelet=wavelet_filter, levels=level_decomp)
    Wstar_w_perm = tf.transpose(Wstar_w[np.newaxis,...], perm = [3,1,2,0])

    # Apply Radon Transform
    RWstar_w =  odl_op_layer(Wstar_w_perm)
    RstarRWstar_w = odl_op_layer_adjoint(RWstar_w)
    RstarRWstar_w_rescaled = rescal * RstarRWstar_w

    # Go back to wavelet domain
    RstarRWstar_w_rescaled_perm = tf.transpose(tf.squeeze(RstarRWstar_w_rescaled,[3]), perm=[1,2,0])
    WRstarRWstar_w = tfw.dwt2d(RstarRWstar_w_rescaled_perm, wavelet=wavelet_filter, levels=level_decomp)
    
    return tf.transpose(WRstarRWstar_w, perm = [2,0,1])[...,np.newaxis]

        
def Get_Rescal_Factor(size_image, angles, method, level_decomp, np_prec ='', tf_prec='',\
                      precision_float='',nbDetectorPixels='', wavelet_type=''):
    """
    Computes the inverse of an approximation of the norm of R*R
    Parameters
    -----------
        size_image (int)           : object dimension (multiple of 2)
        angles(numpy array)        : angles of the 'seen' projections
        method (string)            : 'scikit', 'bowtie' or 'astra' depending of the way R*R was implemented
        level_decomp (int)         : number of wavelet scales (J-J0-1 in the article)   
        np_prec(type)              : numpy type corresponding to the desired machine precision    
        tf_prec(tensorflow.dtype)  : tensorflow type corresponding to the desired machine precision
        precision_float (int)      : machine precision (16 for half-precision, 
                                    32 for single-precision, 64 for double-precision)               
        nbDetectorPixels (int)     : sinogram width (needed only if method=='astra')
        wavelet_type (string)      : type of wavelets ('haar','db2') (needed only if method=='bowtie')
    Returns
    -----------        
        (float)                    : rescaling factor
    """ 
    num_simulations = 10
    if method=='scikit':
        b_k1 = np.random.rand(size_image,size_image)
        for i in range(num_simulations):
            b_k = b_k1
            b_k1 = iradon(radon(b_k,theta=angles,circle=False),theta=angles,circle=False,filter=None)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k1 = b_k1 / b_k1_norm
        b_k2 = iradon(radon(b_k,theta=angles,circle=False),theta=angles,circle=False,filter=None)

    elif method=='bowtie':
        tf.compat.v1.reset_default_graph()
        wavelet_filter = tfw.create_wavelet_transform_filters(wavelet_type, tf_prec, np_prec) 
        # Creation of the filters
        dict_filters = creation_dict_bowties(size_image, level_decomp, angles, wavelet_type)
        dict_filters_tf = initialize_filters_F(dict_filters, 0, False, np_prec, tf_prec)   
    
        b_k_plc = tf.compat.v1.placeholder(tf_prec, shape=(1, size_image, size_image, 1))
        b_k1 = np.random.rand(1, size_image,size_image, 1)   
        
        # Go to wavelet domain
        Wb_k_plc = tf.transpose(tf.squeeze(b_k_plc,[3]), perm=[1,2,0])
        Wb_k_plc = tfw.dwt2d(Wb_k_plc, wavelet=wavelet_filter, levels=level_decomp)
        Wb_k_plc = tf.transpose(Wb_k_plc, perm=[2,0,1])[...,np.newaxis]
        #Apply bowtie filters
        Wb_k1_plc = ConvFilters(Wb_k_plc, size_image, level_decomp, dict_filters_tf, 1, tf_prec)
        # Go back to image domain
        b_k1_plc = tf.transpose(tf.squeeze(Wb_k1_plc,[3]), perm=[1,2,0])
        b_k1_plc = tfw.idwt2d(b_k1_plc, wavelet=wavelet_filter, levels=level_decomp)
        b_k1_plc = tf.transpose(b_k1_plc, perm=[2,0,1])[...,np.newaxis]
        
        init = tf.compat.v1.global_variables_initializer()
        
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for i in range(num_simulations):
                    b_k = b_k1
                    b_k1 = sess.run(b_k1_plc, feed_dict={b_k_plc:b_k})
                    b_k1_norm = np.linalg.norm(b_k1)
                    b_k1 = b_k1 / b_k1_norm   
            b_k2 = sess.run(b_k1_plc, feed_dict={b_k_plc:b_k})
            sess.close() 
            
        b_k1 = b_k1[0,:,:,0] 
        b_k2 = b_k2[0,:,:,0] 
        return np.sum(b_k1 * b_k1)/np.sum(b_k2 * b_k1) 

    elif method=='astra':
        precision_float_str = 'float'+str(precision_float)
        
        tf.compat.v1.reset_default_graph()
        space = odl.uniform_discr([-size_image//2, -size_image//2], [size_image//2, size_image//2],
                                  [size_image, size_image], dtype=precision_float_str)
        angle_partition = odl.uniform_partition(math.radians(angles[0]+90),
                                  math.radians(angles[-1]+90), angles.shape[0])
        
        detector_partition = odl.uniform_partition(-nbDetectorPixels//2, nbDetectorPixels//2, nbDetectorPixels)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        
        ray_transform = odl.tomo.RayTransform(space, geometry, impl = 'astra_cuda')
        
        odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(ray_transform,
                                                                  'RayTransform')
        odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(ray_transform.adjoint,
                                                                               'RayTransformAdjoint')

        u = tf.placeholder(tf_prec, shape=(1, size_image, size_image, 1))
        Ru =  odl_op_layer(u)
        RstarRu= odl_op_layer_adjoint(Ru) 
        b_k1 = np.random.rand(1,size_image,size_image,1)
        init = tf.compat.v1.global_variables_initializer()
        
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for i in range(num_simulations):
                    b_k = b_k1
                    b_k1 = sess.run(RstarRu, feed_dict={u:b_k})
                    b_k1_norm = np.linalg.norm(b_k1)
                    b_k1 = b_k1 / b_k1_norm   
            b_k2 = sess.run(RstarRu, feed_dict={u:b_k})
            sess.close() 
         
        b_k1 = b_k1[0,:,:,0] 
        b_k2 = b_k2[0,:,:,0] 
    return np.sum(b_k1 * b_k1)/np.sum(b_k2 * b_k1) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def PSIDONetF(wave_bp, size_image, level_decomp, angles, dictionaries, thetas, \
            alphas, betas, filter_size, nb_unrolledBlocks, nb_repetBlock, wavelet_type, \
            wavelet_filter, nbDetectorPixels, rescal, tf_prec, np_prec, precision_float):
    """
    Builds the graph of PsiDONet-F
        r_k+1 = w_k + alpha_k * (WRstar m - beta_k * (K0(w_k) + K1(w_k))) 
        w_k+1 = S_theta_k(r_k+1)
    Parameters
    -----------
        wave_bp (tensor)           : wavelet representation of the back-projection limited-angle image, shape b*h*w*1
        size_image (int)           : object dimension (multiple of 2)
        level_decomp (int)         : number of wavelet scales (J-J0-1 in the article) 
        angles(numpy array)        : angles of the 'seen' projections  
        dictionaries(list)         : list of trainable dictionaries (tensorflow variables) 
        thetas(list)               : list of trainable theta parameters (tensorflow variables) 
        alphas(list)               : list of trainable alpha parameters (tensorflow variables) 
        betas(list)                : list of trainable beta parameters (tensorflow variables)  
        filter_size (int)          : size of every trainable filter  
        nb_unrolledBlocks (int)    : number of different set of trainable parameters 
        nb_repetBlock (int)        : number of times each set of trainable parameters is used   
        wavelet_type (string)      : type of wavelets
        wavelet_filter(tfw.Wavelet): wavelet coefficients for the computation of the wavelet transform        
        nbDetectorPixels (int)     : sinogram width 
        rescal (int)               : rescaling factor so that the normal operator of the radon transform has unit norm
        tf_prec(tensorflow.dtype)  : tensorflow type corresponding to the desired machine precision
        np_prec(type)              : numpy type corresponding to the desired machine precision 
        precision_float (int)      : machine precision (16 for half-precision, 
                                     32 for single-precision, 64 for double-precision)
    Returns
    -----------        
        (tensor)                  : predicted wavelet object, shape b*h*w*1
    """   
    # Creation of the bowties to initialize the fixed filters
    dict_filters = creation_dict_bowties(size_image, level_decomp, angles, wavelet_type)
    # Separation between the fixed operator (border of the filters) and the learnt operator (center of the filters)
    dict_hole, _ = separate_dict_hole_NN(dict_filters, level_decomp, filter_size)
    dict_hole = initialize_filters_F(dict_hole, 0, False, np_prec, tf_prec) 

    W_i = tf.zeros_like(wave_bp)
     
    for i in range(nb_unrolledBlocks):
        for j in range(nb_repetBlock):
            print('unrolled block: '+ str(i+1) + ' -- iteration: ' + str(j+1))
            # convolutions with fixed bowties
            K0 = ConvFilters(W_i, size_image, level_decomp,\
                                         dict_hole, rescal, tf_prec)
            
            # convolutions with smaller bowties to be learnt
            K1 = ConvFilters(W_i, size_image, level_decomp, dictionaries[i], rescal, tf_prec)
            # ista iteration
            r = tf.add(W_i, alphas[i]*tf.subtract(wave_bp,betas[i]*tf.add(K0,K1)))
            # soft-thresholding
            W_i = tf.multiply(tf.sign(r), tf.nn.relu(tf.abs(r) - thetas[i]))
            
    return W_i

def PSIDONetFplus(wave_bp, size_image, level_decomp, angles, dictionaries, thetas, \
            alphas, betas, filter_size, nb_unrolledBlocks, nb_repetBlock, wavelet_type,  \
            wavelet_filter, nbDetectorPixels, rescal, tf_prec, np_prec, precision_float):
    """
    Builds the graph of PsiDONet-F+
        r_k+1 = w_k + alpha_k * (WRstar m - beta_k * (K0(w_k) + K1(w_k))) 
        w_k+1 = S_{10^theta_k}(r_k+1)
    Parameters
    -----------
        wave_bp (tensor)           : wavelet representation of the back-projection limited-angle image, shape b*h*w*1
        size_image (int)           : object dimension (multiple of 2)
        level_decomp (int)         : number of wavelet scales (J-J0-1 in the article) 
        angles(numpy array)        : angles of the 'seen' projections  
        dictionaries(list)         : list of trainable dictionaries (tensorflow variables) 
        thetas(list)               : list of trainable theta parameters (tensorflow variables) 
        alphas(list)               : list of trainable alpha parameters (tensorflow variables) 
        betas(list)                : list of trainable beta parameters (tensorflow variables)  
        filter_size (int)          : size of every trainable filter  
        nb_unrolledBlocks (int)    : number of different set of trainable parameters 
        nb_repetBlock (int)        : number of times each set of trainable parameters is used   
        wavelet_type (string)      : type of wavelets
        wavelet_filter(tfw.Wavelet): wavelet coefficients for the computation of the wavelet transform        
        nbDetectorPixels (int)     : sinogram width 
        rescal (int)               : rescaling factor so that the normal operator of the radon transform has unit norm
        tf_prec(tensorflow.dtype)  : tensorflow type corresponding to the desired machine precision
        np_prec(type)              : numpy type corresponding to the desired machine precision 
        precision_float (int)      : machine precision (16 for half-precision, 
                                     32 for single-precision, 64 for double-precision)
    Returns
    -----------        
        (tensor)                  : predicted wavelet object, shape b*h*w*1
    """
    # Creation of the bowties to initialize the fixed filters
    dict_filters = creation_dict_bowties(size_image, level_decomp, angles, wavelet_type)
    # Separation between the fixed operator (border of the filters) and the learnt operator (center of the filters)
    dict_hole, _ = separate_dict_hole_NN(dict_filters, level_decomp, filter_size)
    dict_hole = initialize_filters_F(dict_hole, 0, False, np_prec, tf_prec) 

    W_i = tf.zeros_like(wave_bp)
     
    for i in range(nb_unrolledBlocks):
        for j in range(nb_repetBlock):
            # convolutions with fixed bowties
            K0 = ConvFilters(W_i, size_image, level_decomp,\
                                         dict_hole, rescal, tf_prec)
            
            # convolutions with smaller bowties to be learnt
            K1 = ConvFilters(W_i, size_image, level_decomp, dictionaries[i], rescal, tf_prec)
            # ista iteration
            r = tf.add(W_i, alphas[i]*tf.subtract(wave_bp,betas[i]*tf.add(K0,K1)))
            # soft-thresholding
            W_i = tf.multiply(tf.sign(r), tf.nn.relu(tf.abs(r) - tf.math.pow(np_prec(10),thetas[i])))
            
    return W_i

def PSIDONetO(wave_bp, size_image, level_decomp, angles, dictionaries, thetas, \
            alphas, betas, filter_size, nb_unrolledBlocks, nb_repetBlock, wavelet_type, \
            wavelet_filter, nbDetectorPixels, rescal, tf_prec, np_prec, precision_float):
    """
    Builds the graph of PsiDONet-O
        r_k+1 = w_k + alpha_k * (WRstar m - WRstarRWstar(w_k)) +  beta_k * NN_w(w_k) 
        w_k+1 = S_theta_k(r_k+1)
    Parameters
    -----------
        wave_bp (tensor)           : wavelet representation of the back-projection limited-angle image, shape b*h*w*1
        size_image (int)           : object dimension (multiple of 2)
        level_decomp (int)         : number of wavelet scales (J-J0-1 in the article) 
        angles(numpy array)        : angles of the 'seen' projections  
        dictionaries(list)         : list of trainable dictionaries (tensorflow variables) 
        thetas(list)               : list of trainable theta parameters (tensorflow variables) 
        alphas(list)               : list of trainable alpha parameters (tensorflow variables) 
        betas(list)                : list of trainable beta parameters (tensorflow variables)  
        filter_size (int)          : size of every trainable filter  
        nb_unrolledBlocks (int)    : number of different set of trainable parameters 
        nb_repetBlock (int)        : number of times each set of trainable parameters is used   
        wavelet_type (string)      : type of wavelets
        wavelet_filter(tfw.Wavelet): wavelet coefficients for the computation of the wavelet transform        
        nbDetectorPixels (int)     : sinogram width 
        rescal (int)               : rescaling factor so that the normal operator of the radon transform has unit norm
        tf_prec(tensorflow.dtype)  : tensorflow type corresponding to the desired machine precision
        np_prec(type)              : numpy type corresponding to the desired machine precision 
        precision_float (int)      : machine precision (16 for half-precision, 
                                     32 for single-precision, 64 for double-precision)
    Returns
    -----------        
        (tensor)                  : predicted wavelet object, shape b*h*w*1
    """    
    W_i = tf.zeros_like(wave_bp)
     
    for i in range(nb_unrolledBlocks):    
        for j in range(nb_repetBlock):
            print('unrolled block: '+ str(i+1) + ' -- iteration: ' + str(j+1))
            # convolutions with fixed bowties
            WRstarRWstar_w = WRstarRWstar(W_i, size_image, wavelet_filter, level_decomp, angles,\
                                          nbDetectorPixels, rescal, tf_prec, precision_float)
            # convolutions with smaller bowties to be learnt
            NN_w = ConvFilters(W_i, size_image, level_decomp, dictionaries[i], 1, tf_prec)
            # ista iteration
            r = tf.add(W_i, tf.add(alphas[i]*tf.subtract(wave_bp,WRstarRWstar_w), betas[i]*NN_w))
            # soft-thresholding
            W_i = tf.multiply(tf.sign(r), tf.nn.relu(tf.abs(r) - thetas[i]))

    return  W_i

def PSIDONetOplus(wave_bp, size_image, level_decomp, angles, dictionaries, thetas, \
            alphas, betas, filter_size, nb_unrolledBlocks, nb_repetBlock, wavelet_type, \
            wavelet_filter, nbDetectorPixels, rescal, tf_prec, np_prec, precision_float):
    """
    Builds the graph of PsiDONet-O+
        r_k+1 = w_k + alpha_k * (WRstar m - WRstarRWstar(w_k)) +  beta_k * NN_w(w_k) 
        w_k+1 = S_{10^theta_k}(r_k+1)
    Parameters
    -----------
        wave_bp (tensor)           : wavelet representation of the back-projection limited-angle image, shape b*h*w*1
        size_image (int)           : object dimension (multiple of 2)
        level_decomp (int)         : number of wavelet scales (J-J0-1 in the article) 
        angles(numpy array)        : angles of the 'seen' projections  
        dictionaries(list)         : list of trainable dictionaries (tensorflow variables) 
        thetas(list)               : list of trainable theta parameters (tensorflow variables) 
        alphas(list)               : list of trainable alpha parameters (tensorflow variables) 
        betas(list)                : list of trainable beta parameters (tensorflow variables)  
        filter_size (int)          : size of every trainable filter  
        nb_unrolledBlocks (int)    : number of different set of trainable parameters 
        nb_repetBlock (int)        : number of times each set of trainable parameters is used   
        wavelet_type (string)      : type of wavelets
        wavelet_filter(tfw.Wavelet): wavelet coefficients for the computation of the wavelet transform        
        nbDetectorPixels (int)     : sinogram width 
        rescal (int)               : rescaling factor so that the normal operator of the radon transform has unit norm
        tf_prec(tensorflow.dtype)  : tensorflow type corresponding to the desired machine precision
        np_prec(type)              : numpy type corresponding to the desired machine precision 
        precision_float (int)      : machine precision (16 for half-precision, 
                                     32 for single-precision, 64 for double-precision)
    Returns
    -----------        
        (tensor)                  : predicted wavelet object, shape b*h*w*1
    """ 
    W_i = tf.zeros_like(wave_bp)
     
    for i in range(nb_unrolledBlocks):    
        for j in range(nb_repetBlock):
            print('unrolled block: '+ str(i+1) + ' -- iteration: ' + str(iter+1))
            # Block i
            # convolutions with fixed bowties
            WRstarRWstar_w = WRstarRWstar(W_i, size_image, wavelet_filter, level_decomp, angles,\
                                          nbDetectorPixels, rescal, tf_prec, precision_float)
            # convolutions with smaller bowties to be learnt
            NN_w = ConvFilters(W_i, size_image, level_decomp,  dictionaries[i], 1, tf_prec)
            # ista iteration
            r = tf.add(W_i, tf.add(alphas[i]*tf.subtract(wave_bp,WRstarRWstar_w), betas[i]*NN_w))
            # soft-thresholding
            W_i = tf.multiply(tf.sign(r), tf.nn.relu(tf.abs(r) - tf.math.pow(np_prec(10),thetas[i])))

    return  W_i






