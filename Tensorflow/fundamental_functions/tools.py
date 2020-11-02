# -*- coding: utf-8 -*-
"""
Tools.
Tensorflow version.

Functions
-----------
        compute_angles           :    computes the array of 'seen' angles 
        dtypes                   :    returns the tensorflow type and the numpy type to be used 
                                      depending on the chosen machine precision
        LoadData                 :    loads the ground truths and corresponding limited-angle back-projections
        get_nbDetectorPixels     :    returns the width of the sinogram, that is the number of 
                                      sensors measuring each projection
        compute_quality_results  :    computes the mean quality assessments of the restored images                        
                            
@author: Mathilde Galinier
@date: 29/09/2020
"""

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from skimage.measure import compare_ssim
from skimage.transform import iradon
import pywt
from haar_psi import haar_psi_numpy

def compute_angles(alpha, step):
    """
    Computes the array of angles 'seen' by the Radon transform.
    Parameters
    -----------
        alpha (int)         : missing angle on the first quadrant
        step  (int)         : number of degrees between two consecutive projections (>=1). 
    Returns
    -----------        
       angles(numpy array) : array of floats consisting of the angles of each measured projection 
    """
    first_angle = alpha
    last_angle = 179.-alpha
    Nangles = int(np.ceil((last_angle-first_angle)/float(step)))+1
    angles = (np.linspace(first_angle, last_angle, Nangles, endpoint=True)).astype(int)
    return angles             
                
def dtypes(precision_float):
    """
    Returns the tensorflow type and the numpy type to be used depending on the chosen machine precision
    Parameters
    -----------
        precision_float (int)   : machine precision (16 for half-precision, 
                                  32 for single-precision, 64 for double-precision)
    Returns
    -----------        
       (tensorflow.dtype)       : tensorflow type corresponding to the desired machine precision 
       (type)                   : numpy type corresponding to the desired machine precision 
    """
    return tf.as_dtype('float'+str(precision_float)), getattr(np,'float'+str(precision_float))

def LoadData(angles, indices_samples, size_image, level_decomp, type_data, wavelet_type, rescal_scikit,\
             folder, np_prec): 
    """
    Loads the ground truths and corresponding limited-angle back-projection reconstructions.
    They can be returned as objects in the image domain, or wavelet domain, or both of them.
    Parameters
    -----------
        angles (numpy array)          :  array containing all the 'seen' angles
        indices_samples (numpy array) :  indices of the images to be loaded
        size_image (int)              :  dimension of the (square) images 
        level_decomp (int)            :  number of wavelet scales (J-J0-1 in the article)
        type_data (string)            :  defines in which domain the objects must be returned 
                                         ('im', 'wave', or 'both')
        wavelet_type (string)         :  type of wavelets    
        rescal_scikit (float)         :  a rescaling factor must be used so that the normal radon operator
                                         has a unit norm. Such a coefficient can be computed with our 
                                         function Get_Rescal_Factor. 
        folder (string)               :  folder where to find the dataset of interest
        np_prec (type)                :  numpy type corresponding to the desired machine precision                       
    Returns
    -----------        
       names (list)                   :  names of the loaded images   
       im_bp (numpy array)            :  loaded back-projections in the image domain, shape b*h*w*1
       im_true (numpy array)          :  loaded ground truths in the image domain, shape b*h*w*1
       wave_bp (numpy array)          :  loaded back-projections in the wavelet domain, shape b*h*w*1
       wave_true (numpy array)        :  loaded ground truths in the wavelet domain, shape b*h*w*1
    """
    # List of the folders of interest
    folder_im        = os.path.join(folder,'Images')
    file_names_im    = sorted(os.listdir(folder_im))
    file_list_im     = [os.path.join(folder_im, i) for i in file_names_im]
            
    folder_sino      = os.path.join(folder,'Sinograms')
    file_names_sino  = sorted(os.listdir(folder_sino))
    file_list_sino   = [os.path.join(folder_sino, i) for i in file_names_sino]
    
    # Crop width in order to get the right image dimensions
    # Those lines of code can be removed or modified depending on how the dataset has been generated
    if size_image==64 or size_image==256 or size_image==512:
        shift_abs = 2
    elif size_image==128:
        shift_abs = 1
    
    dataset_size = len(indices_samples)
    # Array initialization    
    if type_data=='im' or type_data=='both':    
        im_bp = np.zeros((dataset_size, size_image, size_image,1), dtype = np_prec)
        im_true = np.zeros((dataset_size, size_image, size_image,1), dtype = np_prec)       
    if type_data=='wave' or type_data=='both':
        wave_bp = np.zeros((dataset_size, size_image, size_image,1), dtype = np_prec)
        wave_true = np.zeros((dataset_size, size_image, size_image,1), dtype = np_prec)
    
    names = []
    # Loop over the samples under consideration
    for i, index in enumerate(indices_samples):
        input_sino = sio.loadmat(file_list_sino[index])['mnc']
        input_sino = input_sino[:,angles]
        input_sino[input_sino<0]=0

        im_bp_index    = iradon(input_sino, theta=angles, circle=False, filter=None)[shift_abs:-1,shift_abs:-1] * rescal_scikit
        im_true_index  = sio.loadmat(file_list_im[index])['im_reduced']

        names.append(file_names_im[index])
        if type_data=='im' or type_data=='both': 
            im_bp[i,:,:,:] = im_bp_index[...,np.newaxis]
            im_true[i,:,:,:] = im_true_index[...,np.newaxis]
        if type_data=='wave' or type_data=='both': 
            wave_bp_index, _   = pywt.coeffs_to_array(pywt.wavedec2(im_bp_index, wavelet_type, level=level_decomp, mode = 'periodization'))
            wave_true_index, _ = pywt.coeffs_to_array(pywt.wavedec2(im_true_index, wavelet_type, level=level_decomp, mode = 'periodization'))

            wave_bp[i,:,:,:] = wave_bp_index[...,np.newaxis]
            wave_true[i,:, :,:] = wave_true_index[...,np.newaxis]
        
    if type_data=='im':
        return names, im_bp, im_true
    elif type_data=='wave':
        return names, wave_bp, wave_true
    elif type_data=='both':
        return names, im_bp, im_true, wave_bp, wave_true
        
def get_nbDetectorPixels(path_datasets):
    """
    Gets the width of the sinogram, that is the number of sensors measuring each projection
    Parameters
    -----------
        path_datasets (string)  : folder of the dataset
    Returns
    -----------        
       (int)                    : width of a sinogram of the dataset
    """
    folder_sino        = os.path.join(path_datasets,'train','Sinograms')
    file_names_sino      = sorted(os.listdir(folder_sino))
    return sio.loadmat(os.path.join(folder_sino,file_names_sino[0]))['mnc'].shape[0]

def compute_quality_results(path_true, path_restored, precision_float):
    """
    Computes and saves the mean quality assessments of the restored images.  
    Parameters
    -----------
        path_true (string)      : ground truth folder
        path_restored (string)  : restored image folder
        precision_float (int)   : machine precision (16 for half-precision, 
                                  32 for single-precision, 64 for double-precision)
    Returns
    -----------        
       (float)                  : average relative error of the restored images
       (float)                  : average MSE of the restored images
       (float)                  : average SSIM of the restored images
       (float)                  : average PSNR of the restored images   
       (float)                  : average HaarPSI of the restored images       
    """
    file_names       = os.listdir(path_restored)
    file_list       = [[os.path.join(path_true, i),
                        os.path.join(path_restored,i)] for i in file_names]

    # initialization of vectors containing the quality results
    MSE_tab          = np.zeros(len(file_names))
    relative_err_tab = np.zeros(len(file_names))
    SSIM_tab         = np.zeros(len(file_names))
    PSNR_tab         = np.zeros(len(file_names))
    HaarPSI_tab      = np.zeros(len(file_names))
    
    for i in range(0,len(file_names)):
        # load images
        x_true          = sio.loadmat(file_list[i][0])['im_reduced'].astype('float'+str(precision_float))
        x_psidonet      = sio.loadmat(file_list[i][1])['image']
        
        # compute ssim
        relative_err_tab[i] = np.linalg.norm(x_true - x_psidonet)/np.linalg.norm(x_true)
        MSE_tab[i]          = np.mean((x_true - x_psidonet)**2)
        SSIM_tab[i]         = compare_ssim(x_true, x_psidonet, data_range=1, multichannel=False)
        PSNR_tab[i]         = -10*np.log10(np.mean((x_true-x_psidonet)**2))
        HaarPSI_tab[i]      =  haar_psi_numpy(x_true*255,x_psidonet*255)[0]
        
    relative_err_mean = np.mean(relative_err_tab)
    MSE_mean          = np.mean(MSE_tab)
    SSIM_mean         = np.mean(SSIM_tab)
    PSNR_mean         = np.mean(PSNR_tab)
    HaarPSI_mean      = np.mean(HaarPSI_tab)
    
    print('Quality assessment of the restaured images:')
    print('Average Relative error: %.3f'%(relative_err_mean))
    print('Average MSE           : %.3f'%(MSE_mean))
    print('Average SSIM          : %.3f'%(SSIM_mean))
    print('Average PSNR          : %.3f'%(PSNR_mean))
    print('Average HaarPSI       : %.3f'%(HaarPSI_mean))
  
    #Write mean relative error in file
    file = open(path_restored + '/mean_relative_error_' + str(len(file_names)) + 'ex.txt','w')
    file.write('Relative error (mean over set): ' + str(relative_err_mean) +'\n')
    file.write('MSE (mean over set): ' + str(MSE_mean) +'\n')
    file.write('SSIM (mean over set): ' + str(SSIM_mean) +'\n')
    file.write('PSNR  (mean over set): ' + str(PSNR_mean) +'\n')
    file.write('HaarPSI (mean over set): ' + str(HaarPSI_mean))    
    file.close()       
    return relative_err_mean, MSE_mean, SSIM_mean, PSNR_mean, HaarPSI_mean
