# -*- coding: utf-8 -*-
"""
Tools.
PyTorch version.

Functions
-----------
        compute_angles           :    computes the array of 'seen' angles 
        torch_dtype              :    returns the torch type and the cuda type to be used 
                                      depending on the chosen machine precision
        Get_Rescal_Factor        :                                  
        get_nbDetectorPixels     :    returns the width of the sinogram, that is the number of 
                                      sensors measuring each projection
        compute_quality_results  :    computes the mean quality assessments of the restored images                        
                            
@author: Mathilde Galinier
@date: 29/09/2020
"""

import numpy as np
import torch 
import scipy.io as sio
from skimage.measure import compare_ssim
import os
from sklearn.metrics import mean_squared_error
from fundamental_functions.haar_psi import haar_psi_numpy

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
    Nangles = int(np.ceil((last_angle-first_angle)/step))+1
    angles = (np.linspace(first_angle, last_angle, Nangles, endpoint=True)).astype(int)
    return angles             
                
def torch_dtype(precision_float):
    """
    Returns the torch type and the cuda type to be used depending on the chosen machine precision
    Parameters
    -----------
        precision_float (int)   : machine precision (16 for half-precision, 
                                  32 for single-precision, 64 for double-precision)
    Returns
    -----------        
       (torch.dtype)       : torch type corresponding to the desired machine precision 
       (torch.tensortype)  : cuda type corresponding to the desired machine precision 
    """    
    if precision_float == 16:
      return torch.float16, torch.cuda.HalfTensor
    elif precision_float == 32:
      return torch.float32, torch.cuda.FloatTensor
    elif precision_float == 64:
      return torch.float64, torch.cuda.DoubleTensor

def Get_Rescal_Factor(dim_inputs, transf=None, type_input='numpy'):
    """
    Computes the inverse of an approximation of the norm of R*R
    Parameters
    -----------
        dim_inputs(numpy array)    : array specifying the dimensions of the input of transf
        transf(method)             : operator R*R
        type_input(string)         : 'numpy' or 'tensor' according to the type of the input of transf
    Returns
    -----------        
        (float)                    : rescaling factor
    """ 
    num_simulations = 10
    if type_input=='numpy':
        b_k1 = np.random.rand(*dim_inputs) 
    elif type_input == 'tensor':
        b_k1 = torch.rand(*dim_inputs).cuda()
        
    for i in range(num_simulations):
        b_k = b_k1
        b_k1 = transf(b_k)
        if type_input=='numpy':
            b_k1_norm = np.linalg.norm(b_k1)
        elif type_input == 'tensor':
            b_k1_norm = torch.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm
    b_k2 = transf(b_k)
    if type_input=='numpy':
        norm = np.sum(b_k2 * b_k1)/np.sum(b_k1 * b_k1) 
    elif type_input == 'tensor':
        norm = (torch.sum(b_k2 * b_k1)/torch.sum(b_k1 * b_k1)).item()
    return 1/norm 

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

    ### initialization of vectors containing the quality results
    MSE_tab          = np.zeros(len(file_names))
    relative_err_tab = np.zeros(len(file_names))
    SSIM_tab         = np.zeros(len(file_names))
    PSNR_tab         = np.zeros(len(file_names))
    HaarPSI_tab      = np.zeros(len(file_names))
    
    for i in range(0,len(file_names)):
        ### load images
        x_true          = sio.loadmat(file_list[i][0])['im_reduced'].astype('float'+str(precision_float))
        x_psidonet      = sio.loadmat(file_list[i][1])['image']
        
        ### compute ssim
        relative_err_tab[i] = np.linalg.norm(x_true - x_psidonet)/np.linalg.norm(x_true)
        MSE_tab[i]          = mean_squared_error(x_true, x_psidonet)
        SSIM_tab[i]         = compare_ssim(x_true, x_psidonet, data_range=1, multichannel=False)
        PSNR_tab[i]         = -10*np.log10(np.mean((x_true-x_psidonet)**2))
        HaarPSI_tab[i]      =  haar_psi_numpy(x_true*255,x_psidonet*255)[0]
        
    print(relative_err_tab)
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
    file = open(path_restored + '/../mean_relative_error_' + str(len(file_names)) + 'ex.txt','w')
    file.write('Relative error (mean over test set): ' + str(relative_err_mean) +'\n')
    file.write('MSE (mean over test set): ' + str(MSE_mean) +'\n')
    file.write('SSIM (mean over test set): ' + str(SSIM_mean) +'\n')
    file.write('PSNR  (mean over test set): ' + str(PSNR_mean) +'\n')
    file.write('HaarPSI (mean over test set): ' + str(HaarPSI_mean))    
    file.close()       
    return relative_err_mean, MSE_mean, SSIM_mean, PSNR_mean, HaarPSI_mean