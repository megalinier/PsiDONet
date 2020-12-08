# -*- coding: utf-8 -*-
"""
Modules.
PyTorch version.

Functions
-----------
        compute_PSNR              :   Computes the average peak signal-to-noise ratio of a list of images.
        compute_PSNR_SSIM         :   Comptues the average PSNR and SSIM of a list of images.
        OpenMat                   :   Converts a numpy array loaded from a .mat file into a properly ordered tensor.  
        Reorg_wave_coeffs         :   Reorganizes the wavelet coefficients included in the pytorch_wavelets coefficients 
                                      as a single array.    
        Reorg_wave_coeffs_inverse :   Reorganizes a single array into list of wavelets coefficients.
Classes
-----------
        OpenMat_transf            :   Transforms an array into an ordered tensor. 
        Missing_angle_sino        :   Returns sinogram with missing angles from complete-angle sinogram.
        Scikit_transf             :   Class for the direct and inverse Radon transform from the scikit package.  
        Astra_transf              :   Class for the direct and inverse Radon transform from the Astra-Toolbox.
        Reorg_wave_transf         :   Class for applying Reorg_wave_coeffs.  
        Reorg_wave_inverse_transf :   Class for applying Reorg_wave_coeffs_inverse.   
        MyDataset                 :   Loads and transforms images before feeding it to the network.  
                    
@author: Mathilde Galinier
@date: 07/12/2020
"""
import numpy as np
import math
import os
import scipy.io as sio
import torch
from skimage.transform import radon, iradon
from torchvision import transforms
import odl
from odl.contrib.torch import OperatorModule
import PyTorch_ssim as compute_SSIM

def compute_PSNR(x_true, x):
    """
    Computes the average peak signal-to-noise ratio of a list of images.
    Parameters
    ----------
        x_true (torch.FloatTensor): ground-truth images, size b*c*h*w 
	    x      (torch.FloatTensor): restored images, size b*c*h*w 
    Returns
    -------
        (torch.FloatTensor): average PSNR of a list of images (x^(i))_{1<i<n} expressed in dB
            -10*sum_{i=1}^n(log_10(||x_true^(i)-x^(i)||^2/(c*h*w)))
    """
    return -10*torch.mean(torch.log10(torch.mean(torch.mean(torch.mean((x_true-x)**2,1),1),1)))

def compute_PSNR_SSIM(x_true, x_before, x_after, size_set):
    """
    Computes the average peak signal-to-noise ratio and structural similarity measure of a 
    list of images scaled by the batch size over the total number of images.
    Parameters
    ----------
       	x_true   (Variable): ground-truth images, data of size b*c*h*w
	    x_before (Variable): blurred images, data of size b*c*h*w
	    x_after  (Variable): restored images, data of size b*c*h*w
	    size_set      (int): total number of images
    Returns
    -------
       	(numpy array): PSNR and SSIM values before and after restoration, size 2*2
    """
    size_batch  = x_true.data.size()[0]
    snr_before  = compute_PSNR(x_true.data.cpu(), x_before.data.cpu())* size_batch/size_set
    snr_after   = compute_PSNR(x_true.data.cpu(), x_after.data.cpu())* size_batch/size_set
    ssim_before = torch.Tensor.item(compute_SSIM.ssim(x_before, x_true))* size_batch/size_set
    ssim_after  = torch.Tensor.item(compute_SSIM.ssim(x_after, x_true))* size_batch/size_set
    return np.array((((snr_before),(snr_after)),((ssim_before),(ssim_after))))

def OpenMat(x, precision_type, permute_bool = False):
    """
    Converts a numpy array loaded from a .mat file into a properly ordered tensor.
    Parameters
    ----------
        x (numpy array)       : must be 2D 
        permute_bool(boolean) : must be False for scikit and True for Astra
    Returns
    -------
        (torch.FloatTensor): size b*c*h*w
    """
    if len(x.shape)!=2:
        raise Exception('x must be 2D.')
    else:
        if permute_bool:
            return torch.from_numpy(x).permute(1,0).type(precision_type).unsqueeze(0).unsqueeze(1)
        else:
            return torch.from_numpy(x).type(precision_type).unsqueeze(0).unsqueeze(1)
    
class OpenMat_transf(object):
    """
    Transforms an array into an ordered tensor.
    Attributes
    ----------
        precision_type(torch.dtype) : torch type corresponding to the desired machine precision 
        permute_bool(boolean)       : must be False for scikit and True for Astra
    """
    def __init__(self, precision_type,permute_bool=False):
        super(OpenMat_transf,self).__init__()
        self.precision_type = precision_type
        self.permute_bool   = permute_bool
    def __call__(self,x):
        return OpenMat(x, self.precision_type, self.permute_bool)
    
class Missing_angle_sino(object):
    """
    Returns sinogram with missing angles from complete-angle sinogram.
    Attributes
    ----------
        angles(numpy array): array of floats consisting of the angles of each measured projection 
    """
    def __init__(self, angles):
        super(Missing_angle_sino,self).__init__()
        self.angles = angles
    def __call__(self,sino):
        return sino[:,self.angles]

class Scikit_transf(object):
    """
    Class for the direct and inverse Radon transform from the scikit package.

    Attributes
    ----------
        angles(numpy array): array of floats consisting of the angles of each measured projection
        size_image (int)   : image dimension 
        shift_abs(int)     : number of rows and columns to be removed in order to obtain consistent dimensions
    """
    def __init__(self, angles, size_image):
        super(Scikit_transf,self).__init__()
        self.angles = angles
        if size_image==64 or size_image==256 or size_image==512:
            self.shift_abs = 2
        elif size_image==128:
            self.shift_abs = 1
         
    def R_transf(self,x):
        """
        Computes the direct Radon Transform. (from image to sino)
        Parameters
        ----------
            x(array) : input, size h*w

        Returns
        -------
            R_transf(array) : radon transform of x (Rx), size nbDetectorPixels*Nangles
        """
        R_transf = radon(x, theta=self.angles, circle=False) 
        return R_transf 
           
    def Rstar_transf(self, x):
        """
        Computes the backprojection operator (from sino to image)
        Parameters
        ----------
            x(array) : input, size nbDetectorPixels*Nangles

        Returns
        -------
            Rstar_transf(array) : inverse radon transform of x (Rstar x), size h*w
        """
        Rstar_transf = iradon(x, theta=self.angles, circle=False, filter=None) 
        return Rstar_transf
    
    def Rstar_transf_shift(self, x):
        """
        Computes the backprojection operator (from sino to image)
        As this function is meant to be used on sinograms generated by Matlab,
        we need to remove one or two columns/rows in order to obtain consistent dimensions.
        Parameters
        ----------
            x(array) : input, size nbDetectorPixels*Nangles

        Returns
        -------
            Rstar_transf(array) : inverse radon transform of x (Rstar x), size h*w        
        """
        Rstar_transf = iradon(x, theta=self.angles, circle=False, filter=None)[self.shift_abs:-1,self.shift_abs:-1] 
        return Rstar_transf

    def RstarR_transf(self,x):
        """
        Computes the projection and backprojection operators (from image to image)
        Parameters
        ----------
            x(array) : input, size h*w

        Returns
        -------
            RstarR_transf(array) : direct and inverse radon transforms of x (RstarR x), size h*w        
        """       
        RstarR_transf = transforms.Compose([self.R_transf, self.Rstar_transf])
        return RstarR_transf(x)       
 
class Astra_transf(object):
    """
    Class for the direct and inverse Radon transform from the Astra-Toolbox.
    
    Attributes
    ----------
        space(odl.DiscreteLp)                 : discretised mesh defining the geometry of the object to image
        angle_partition(odl.RectPartition)    : angles of the measured projections
        detector_partition(odl.RectPartition) : defines the detector geometry
        geometry (odl.Parallel2dGeometry)     : defines the parallel beam geometry with flat detector for the data acquisition
        transf(odl.RayTransform)              : direct radon transform 
    """
    def __init__(self, angles, size_image, nbDetectorPixels, precision_float):
        super(Astra_transf,self).__init__()
        self.space = odl.uniform_discr([-size_image//2, -size_image//2], \
                                  [size_image//2, size_image//2],\
                                  [size_image, size_image],\
                                  dtype='float'+str(precision_float))
        # Make a parallel beam geometry with flat detector
        # Angles: uniformly spaced, n = 400, min = 0, max = pi
        self.angle_partition = odl.uniform_partition(math.radians(angles[0]+90),\
                                                math.radians(angles[-1]+90), angles.shape[0])
    
        # Detector: uniformly sampled, n = 400, min = -30, max = 30
        self.detector_partition = odl.uniform_partition(-nbDetectorPixels//2,\
                                                   nbDetectorPixels//2, nbDetectorPixels)
        self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition)
        self.transf = odl.tomo.RayTransform(self.space, self.geometry, impl = 'astra_cuda')   
        
    def R_transf(self, x):
        """
        Computes the direct Radon Transform. (from image to sino)
        Parameters
        ----------
            x(tensor) : input, size b*c*h*w

        Returns
        -------
            R_transf(tensor) : radon transform of x (Rx), size b*c*Nangles*nbDetectorPixels
        """
        ray_transform = self.transf    
        R_transf = OperatorModule(ray_transform)
        return R_transf(x) 
    
    def Rstar_transf(self, x):
        """
        Computes the backprojection operator (from sino to image)
        Parameters
        ----------
            x(array) : input, size b*c*Nangles*nbDetectorPixels

        Returns
        -------
            Rstar_transf(array) : inverse radon transform of x (Rstar x), size b*c*h*w
        """
        ray_transform_adjoint = self.transf.adjoint        
        Rstar_transf = OperatorModule(ray_transform_adjoint)
        return Rstar_transf(x) 

    def RstarR_transf(self,x):
        """
        Computes the projection and backprojection operators (from image to image)
        Parameters
        ----------
            x(array) : input, size b*c*h*w

        Returns
        -------
            RstarR_transf(array) : direct and inverse radon transforms of x (RstarR x), size b*c*h*w        
        """        
        RstarR_transf = transforms.Compose([self.R_transf,self.Rstar_transf])
        return RstarR_transf(x)   
    
def Reorg_wave_coeffs(w, level_decomp, precision_type):
    """
    Reorganizes the wavelet coefficients included in the pytorch_wavelets coefficients wl and wh as a single array.
    
    Parameters
    ----------
        w(list)                     : list of wavelet coefficients
        level_decomp (int)          : number of wavelet scales
        precision_type(torch.dtype) : torch type corresponding to the desired machine precision 
    Returns
    -------
        result(numpy array)         : wavelet coefficients assembled in a single array, size b*c*h*w
    """
    wl, wh = w
    size_block = wl.shape[2]*2**level_decomp
    result = torch.zeros(wl.shape[0], wl.shape[1], size_block, size_block).type(precision_type)
    for l in range(1,level_decomp+1):
        size_block = size_block//2
        result[:,:,:size_block,size_block:2*size_block] = wh[l-1][:,:,1,:,:]
        result[:,:,size_block:2*size_block,:size_block] = wh[l-1][:,:,0,:,:]
        result[:,:,size_block:2*size_block,size_block:2*size_block] = wh[l-1][:,:,2,:,:]
        if l==level_decomp:
            result[:,:,:size_block,:size_block] = wl
    return result       

class Reorg_wave_transf(object):
    """
    Class for applying Reorg_wave_coeffs.
    
    Attributes
    ----------
        level_decomp (int)          : number of wavelet scales
        precision_type(torch.dtype) : torch type corresponding to the desired machine precision     
    """
    def __init__(self, level_decomp, precision_type):
        super(Reorg_wave_transf,self).__init__()
        self.level_decomp = level_decomp
        self.precision_type = precision_type
    def __call__(self, w):
        return Reorg_wave_coeffs(w, self.level_decomp, self.precision_type)
    
def Reorg_wave_coeffs_inverse(w, level_decomp, precision_type):
    """
    Reorganizes the single array w (b*c*H*W) into list of wavelets coefficients wl and wh.
    
    Parameters
    ----------
        w(numpy array)              : wavelet coefficients assembled in a single array, size b*c*h*w
        level_decomp (int)          : number of wavelet scales
        precision_type(torch.dtype) : torch type corresponding to the desired machine precision 
    Returns
    -------
        result(list)                : list of wavelet coefficients
    """
    batch, channel, size_image, _= w.shape
    size_block = size_image//(2**level_decomp)
    wl         = w[:,:,:size_block,:size_block]
    wh         = []
    size_block = size_image
    for l in range(1,level_decomp+1):
        size_block = size_block//2
        wave             = torch.zeros(batch, channel, 3, size_block, size_block).type(precision_type)
        wave[:,:,1,:,:]  = w[:,:,:size_block,size_block:2*size_block]
        wave[:,:,0,:,:]  = w[:,:,size_block:2*size_block,:size_block]
        wave[:,:,2,:,:]  = w[:,:,size_block:2*size_block,size_block:2*size_block]
        wh.append(wave)
    return wl, wh       

class Reorg_wave_inverse_transf(object):
    """
    Class for applying Reorg_wave_coeffs_inverse.
    
    Attributes
    ----------
        level_decomp (int)          : number of wavelet scales
        precision_type(torch.dtype) : torch type corresponding to the desired machine precision     
    """
    def __init__(self, level_decomp, precision_type):
        super(Reorg_wave_inverse_transf,self).__init__()
        self.level_decomp = level_decomp
        self.precision_type = precision_type
    def __call__(self, w):
        return Reorg_wave_coeffs_inverse(w, self.level_decomp, self.precision_type)
    
class MyDataset(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the network.
    Attributes
    ----------
        folder_im       (str)  : path to the folder containing the images
        file_names_im  (list)  : list of strings, list of names of images
        file_list_im   (list)  : list of strings, paths to images
        folder_sino     (str)  : path to the folder containing the images
        file_names_sino(list)  : list of strings, list of names of sinos
        file_list_sino (list)  : list of strings, paths to sinos
        rescaling_factor(float): rescaling factor equal to the inverse of an approximation of the norm of RstarR 
        return_wavelet(boolean): True if the wavelet coefficients corresponding to each image are to be loaded.
        Open_im(method)        : method that is to be used to load the groundtruth images
        Open_BP(method)        : method that is to be used to load the backprojection images
        Open_Wave(method)      : method that is to be used to load the wavelet coefficients
    """
    def __init__(self, folder='/path/to/folder/', Open_im = None, Open_BP = None,\
                 rescaling_factor = 1, return_wavelet = False, Open_Wave = None):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
            folder     (str): path to the folder containing the images (default '/path/to/folder/')
            Open_im(method)        : method that is to be used to load the groundtruth images
            Open_BP(method)        : method that is to be used to load the backprojection images
            rescaling_factor(float): rescaling factor equal to the inverse of an approximation of the norm of RstarR  
            return_wavelet(boolean): True if the wavelet coefficients corresponding to each image are to be loaded.
            Open_Wave(method)      : method that is to be used to load the wavelet coefficients        
        """
        super(MyDataset, self).__init__()
             
        self.folder_im        = os.path.join(folder,'Images')
        self.file_names_im    = sorted(os.listdir(self.folder_im))
        self.file_list_im     = [os.path.join(self.folder_im, i) for i in self.file_names_im]
                
        self.folder_sino      = os.path.join(folder,'Sinograms')
        self.file_names_sino  = sorted(os.listdir(self.folder_sino))
        self.file_list_sino   = [os.path.join(self.folder_sino, i) for i in self.file_names_sino]
        
        self.rescaling_factor = rescaling_factor
        self.return_wavelet   = return_wavelet
        self.Open_im          = Open_im
        self.Open_BP          = Open_BP
        self.Open_Wave        = Open_Wave
        
    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files, should point to .mat
       Returns
       -------
                          (str): image name without the extension
            (torch.FloatTensor): groundtruth image, size c*h*w
            (torch.FloatTensor): backprojection image, size c*h*w
            (torch.FloatTensor): groundtruth wavelet coefficients, size c*h*w (optional)
            (torch.FloatTensor): backprojection wavelet coefficients, size c*h*w (optional)
        """
        im_true   = self.Open_im(sio.loadmat(self.file_list_im[index])['im_reduced'])
        im_bp     = self.Open_BP(sio.loadmat(self.file_list_sino[index])['mnc'])*self.rescaling_factor
        
        if self.return_wavelet:
            wave_true = self.Open_Wave(im_true)
            wave_bp   = self.Open_Wave(im_bp)
            return os.path.splitext(self.file_names_im[index])[0], im_true.squeeze(1), im_bp.squeeze(1), wave_true.squeeze(1), wave_bp.squeeze(1)
        else:
            return os.path.splitext(self.file_names_im[index])[0], im_true.squeeze(1), im_bp.squeeze(1)
    
    def __len__(self):
        return len(self.file_list_im)

