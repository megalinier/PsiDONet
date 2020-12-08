# -*- coding: utf-8 -*-
"""
Model functions.
PyTorch version.

Functions
-----------
            upsampling                :     Unpooling operation.
       
Classes
-----------
            SSIM_loss                 :     Defines the SSIM training loss.
            ConvLearnedFilters        :     Applies the downsampling, upsampling and convolution operations. (cf fig.6 of article)
            ISTAIter                  :     Stands for an unrolled iteration of ISTA in PsiDONet.
            Block                     :     One Block contains nb_repetBlock layers using all the same set of parameters.
            myModel                   :     PsiDONet model. 
            
@author: Mathilde Galinier
@date: 07/12/2020
"""
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import PyTorch_ssim
import torch.nn.functional as F

class SSIM_loss(_Loss):
    """
    Defines the SSIM training loss.
    Attributes
    ----------
        ssim (method): function computing the SSIM
    """
    def __init__(self): 
        super(SSIM_loss, self).__init__()
        self.ssim = PyTorch_ssim.SSIM()
 
    def forward(self, input, target):
        """
        Computes the training loss.
        Parameters
        ----------
      	    input  (torch.FloatTensor): restored images, size b*c*h*w 
            target (torch.FloatTensor): ground-truth images, size b*c*h*w
        Returns
        -------
       	    (torch.FloatTensor): SSIM loss, size 1 
        """
        return 1-self.ssim(input,target)
    
def upsampling(x, stride=2):
    """
    Unpooling operation
    Parameters
    -----------
      x (tensor)            : tensor to be unpooled, shape b*c*h*w
      stride (int)          : power of 2 by which the size of x has to be increased
    Returns
    -----------        
       out (tensor)         : unpooled tensor, shape b*c*2h*2w
    """      
    w = x.new_zeros(stride, stride)
    w[0, 0] = 1
    return F.conv_transpose2d(x, w.expand(1, 1, stride, stride), stride=stride, groups=1)

class ConvLearnedFilters(nn.Module):
    """
    Includes the method for computing the convolutions as shown on Fig. 6 of paper "Deep neural networks 
    for inverse problems with pseudodierential operators: An application to 
    limited-angle tomography"
    by T.Bubba, M. Galinier, L. Ratti, M. Lassas, S. Siltanen, M. Prato 
    
    Attributes
    -----------
    level_decomp (int)           : number of wavelet scales (J-J0-1 in the article)
    size_image (int)             : image dimension (multiple of 2)
    ConvDict(nn.ModuleDict)      : dictionary of convolution operations
    PaddDict(nn.ModuleDict)      : dictionary of padding operations
    DownsamplDict(nn.ModuleDict) : dictionary of downsampling operations
    """
    def __init__(self, filter_size, level_decomp, size_image):
        """
        Parameters
        ----------
        filter_size (int)        : size of each trainable filter
        level_decomp (int)       : number of wavelet scales
        size_image (int)         : image dimension 
        """
        super(ConvLearnedFilters, self).__init__()
        self.level_decomp  = level_decomp
        self.size_image    = size_image
        # Declaring the convolutional filters 
        self.ConvDict      = nn.ModuleDict({})
        self.PaddDict      = nn.ModuleDict({}) 
        for level_pix in range(1,level_decomp+1):
            for position_pix in ['h', 'd', 'v','l']:
                if level_pix < level_decomp and position_pix == 'l':
                    continue
                else:
                    size_block_result = int(size_image/(2**level_pix))
                    # Declaring the Padding operations  
                    key_pad = str(level_pix)+ '_' + position_pix
                    if position_pix == 'v':
                        self.PaddDict['Pad_'+key_pad] = nn.ZeroPad2d((0, size_image-size_block_result,\
                                                                    size_block_result,\
                                                                    size_image-2*size_block_result))
                    elif position_pix == 'h':    
                        self.PaddDict['Pad_'+key_pad] = nn.ZeroPad2d((size_block_result,\
                                                                    size_image-2*size_block_result,\
                                                                    0,size_image-size_block_result))
                    elif position_pix == 'd':  
                        self.PaddDict['Pad_'+key_pad] = nn.ZeroPad2d((size_block_result,\
                                                                    size_image-2*size_block_result,\
                                                                    size_block_result, \
                                                                    size_image-2*size_block_result))
                    elif position_pix == 'l':   
                        self.PaddDict['Pad_'+key_pad] = nn.ZeroPad2d((0,size_image-size_block_result,\
                                                                    0,size_image-size_block_result))
                            
                    for level_filt in range(1,level_decomp+1):
                        for position_filt in ['h', 'd', 'v','l']:
                            if level_filt < level_decomp and position_filt == 'l':
                                continue
                            else:
                                key = key_pad + '_' + str(level_filt)+ '_' + position_filt
                                self.ConvDict['Conv2d_'+ key] = nn.Conv2d(1, 1, kernel_size = filter_size,\
                                                                stride = 1, padding = (filter_size-1)//2,\
                                                                dilation= 1, groups = 1, bias = False,\
                                                                padding_mode = 'zeros')
                
        # Initializing the convolutional filters with xavier initialization                                    
        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                  nn.init.xavier_uniform_(m.weight)

        # Declaring the Downsampling operations
        self.DownsamplDict = nn.ModuleDict({})
        for i in range(1,level_decomp):
            ratio = 2**i
            self.DownsamplDict['Downsamp_' + str(ratio)] = nn.MaxPool2d(ratio)

    def forward(self,w):
        """
        Applies the downsampling, upsampling and convolution operations as shown on fig. 6 of article.

        Parameters
        ----------
         w (tensor)       : wavelet object to which apply the operations, shape b*h*w*1

        Returns
        -------
        NN_w(tensor)      : wavelet object, shape b*h*w*1

        """
        NN_w = torch.zeros_like(w)
        for level_block in range(1, self.level_decomp+1):
            size_block = int(self.size_image/(2**level_block))
            for position_block in ['v','h','d','l']:   
                if level_block<self.level_decomp and position_block=='l':
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
                
                    block_init = w[:,:,indices_pos_x_0:indices_pos_x_0+size_block,\
                                   indices_pos_y_0:indices_pos_y_0+size_block]             
                    
                    for level_filt in range(1, self.level_decomp+1):
                        adapted_block_computed = False
                        for position_filt in ['d','h','v','l']:   
                            if level_filt<self.level_decomp and position_filt=='l':
                                continue
                            else:             
                                ratio_levels = level_block - level_filt
                                if ratio_levels>0 and not(adapted_block_computed): #--> upsampling
                                    block_adapted = upsampling(block_init, 2**ratio_levels)
                                    adapted_block_computed = True
                                elif ratio_levels<0 and not(adapted_block_computed): #--> downsampling
                                    block_adapted = self.DownsamplDict['Downsamp_' + str(2**(-ratio_levels))](block_init)
                                    adapted_block_computed = True
                                elif ratio_levels==0: #--> we keep the same size
                                    block_adapted = block_init
                                    
                                # Convolution with the central part of the filter 
                                key_pad      = str(level_filt)+ '_' + position_filt
                                key          = str(level_block)+ '_' + position_block + '_' +key_pad
                                block_result = self.ConvDict['Conv2d_'+ key](block_adapted)
                                #Padding
                                block_padded = self.PaddDict['Pad_'+key_pad](block_result)
                                NN_w         = NN_w + block_padded                           
        return NN_w        
        
        
class ISTAIter(nn.Module):
    """
    Stands for an unrolled iteration of ISTA in PsiDONet.
    
    Attribues
    ---------
        relu (torch.nn.ReLU): ReLU activation layer
    """
    def __init__(self):
        super(ISTAIter, self).__init__()
        self.relu               = nn.ReLU()
        
    def forward(self, w, wave_bp, alpha, beta, theta, model_unrolling, \
                rescal, RadonWave_transf, ConvLearnedFilters):
        """
        Computes one layer of PsiDONet.
        
        Parameters
        ----------
            w(tensor)                    : wavelet iterate, shape b*h*w*1
            wave_bp(tensor)              : wavelet representation of the backprojection (WRstarm) 
            alpha, beta, theta(Parameter): trainable floats 
            model_unrolling (string)     : name of the version of PsiDONet used to train/test the model
                                           ('PSIDONetO' or 'PSIDONetOplus')
            rescal(float)                : rescaling factor equal to the inverse of an approximation of the norm of RstarR 
            RadonWave_transf(method)     : function computing WRstarRWstar  
            ConvLearnedFilters(class)    : defines the downsampling, upsampling, padding and convolution operations
        
        --> \PsiDONetO
            r_k+1 = w_k + alpha_k * (WRstar m - WRstarRWstar(w_k)) +  beta_k * NN(w_k) 
            w_k+1 = S_theta_k(r_k+1)
            
        Returns
        -------
            (tensor)                     : next iterate in the wavelet domain, shape  b*h*w*1
                  
        """
        # Operators
        WRstarRWstar_w = RadonWave_transf(w)*rescal
        NN_w           = ConvLearnedFilters(w)
        
        #ISTA iteration
        if 'PSIDONetO' in model_unrolling:
            r = w + alpha*(wave_bp - WRstarRWstar_w) + beta*NN_w                                   
            
        return torch.sign(r)*(self.relu(torch.abs(r) - theta)) # Soft-thresholding

class Block(nn.Module):
    """
    One block contains nb_repetBlock layers using all the same set of parameters.
    
    Attributes
    ----------
    nb_repetBlock (int)          : number of times each set of trainable parameters is used 
    ConvLearnedFilters(class)    : defines the downsampling, upsampling, padding and convolution operations
    alpha, beta, theta(Parameter): trainable floats 
    model_unrolling (string)     : name of the version of PsiDONet used to train/test the model
                                           ('PSIDONetO' or 'PSIDONetOplus')
    num_block(int)               : number of the block, i.e. number of the set of trainable parameters used    
    ISTAIter (ISTAIter)          : layer of the model                                    
    """
    def __init__(self, nb_repetBlock, mu, L, level_decomp, filter_size,\
                 model_unrolling, size_image, dtype, num_block):
        """
        Parameters
        ----------
        nb_repetBlock (int)     : number of times each set of trainable parameters is used 
        mu (float)              : standard ISTA regularisation parameter
        L (float)               : standard ISTA constant
        level_decomp (int)      : number of wavelet scales (J-J0-1 in the article)
        filter_size (int)       : size of each trainable filter
        model_unrolling (string): name of the version of PsiDONet used to train/test the model
                                  ('PSIDONetO' or 'PSIDONetOplus')
        size_image (int)        : image dimension
        dtype (torch.tensortype): cuda type corresponding to the desired machine precision      
        num_block(int)          : number of the block, i.e. number of the set of trainable parameters used  
        """
        super(Block, self).__init__()
        self.nb_repetBlock        = nb_repetBlock
        self.ConvLearnedFilters   = ConvLearnedFilters(filter_size, level_decomp, size_image)
        self.alpha                = nn.Parameter(torch.Tensor([1/L]).type(dtype), requires_grad=True)
        self.beta                 = nn.Parameter(torch.Tensor([0]).type(dtype), requires_grad=True) 
        self.model_unrolling      = model_unrolling
        self.num_block            = num_block
        
        if 'plus' in self.model_unrolling:
            self.theta   = nn.Parameter(torch.Tensor([np.log10(mu/L)]).type(dtype), requires_grad=True)
        else:
            self.theta   = nn.Parameter(torch.Tensor([mu/L]).type(dtype), requires_grad=True)

        self.ISTAIter    = ISTAIter()

    def forward(self, w, wave_bp, rescal, RadonWave_transf, save_params, path_save):
        """
        Computes nb_repetBlock successive layers of PsiDONet.
        
        Parameters
        ----------
        w(tensor)                    : current wavelet iterate, shape b*h*w*1
        wave_bp(tensor)              : wavelet representation of the backprojection (WRstarm) 
        rescal(float)                : rescaling factor equal to the inverse of an approximation of the norm of RstarR 
        RadonWave_transf(method)     : function computing WRstarRWstar  
        save_params(boolean)         : if True, parameters alpha, beta and theta are saved
        path_save (string)           : path where to save the trained parameters

        Returns
        -------
        w(tensor)                     : next iterate in the wavelet domain, shape  b*h*w*1

        """
        
        if 'plus' in self.model_unrolling:
            theta = torch.pow(10, self.theta)
        else:
            theta = self.theta
            
        for i in range(self.nb_repetBlock):
            w   = self.ISTAIter(w, wave_bp, self.alpha, self.beta, theta,\
                                self.model_unrolling, rescal, RadonWave_transf, self.ConvLearnedFilters)
        
        if save_params:
            # Write the value of the learned parameters in files
            np.save(os.path.join(path_save,'parameters','block'+str(self.num_block)+'_alpha'),self.alpha.data.cpu())
            np.save(os.path.join(path_save,'parameters','block'+str(self.num_block)+'_beta'),self.beta.data.cpu())
            np.save(os.path.join(path_save,'parameters','block'+str(self.num_block)+'_theta'),theta.data.cpu())
        return w
    
class myModel(nn.Module):
    """
    PsiDONet model
    
    Attributes
    ----------
         RadonWave_transf(method)     : function computing WRstarRWstar 
         Layers(nn.ModuleList)        : layers of PsiDONet
         nb_unrolledBlocks(int)       : number of different sets of trainable parameters 
    """
    def __init__(self, RadonWave_transf, nb_unrolledBlocks, nb_repetBlock, mu, L, level_decomp, filter_size,\
                  model_unrolling, size_image, dtype):
        """
        Parameters
        ----------
            RadonWave_transf(method): function computing WRstarRWstar
            nb_unrolledBlocks(int)  : number of different sets of trainable parameters 
            nb_repetBlock (int)     : number of times each set of trainable parameters is used 
            mu (float)              : standard ISTA regularisation parameter
            L (float)               : standard ISTA constant
            level_decomp (int)      : number of wavelet scales (J-J0-1 in the article)
            filter_size (int)       : size of each trainable filter
            model_unrolling (string): name of the version of PsiDONet used to train/test the model
                                      ('PSIDONetO' or 'PSIDONetOplus')
            size_image (int)        : image dimension
            dtype (torch.tensortype): cuda type corresponding to the desired machine precision      
        """
        super(myModel, self).__init__()    
        
        self.RadonWave_transf  = RadonWave_transf
        self.Layers            = nn.ModuleList()
        self.nb_unrolledBlocks = nb_unrolledBlocks
        
        for i in range(self.nb_unrolledBlocks):
            self.Layers.append(Block(nb_repetBlock, mu, L, level_decomp, filter_size,\
                                      model_unrolling, size_image, dtype, i))
            print('Block------------------- ' +str(i))
        
    def forward(self, w, wave_bp, rescal = 1, save_params=False, path_save=''):
        """
        Computes a reconstruction of the image thanks to the nb_repetBlock*nb_unrolledBlocks layers of PsiDONet.
        
        Parameters
        ----------
            w(tensor)                    : input in the wavelet domain, shape b*h*w*1
            wave_bp(tensor)              : wavelet representation of the backprojection (WRstarm) 
            rescal(float)                : rescaling factor equal to the inverse of an approximation of the norm of RstarR 
            save_params(boolean)         : if True, parameters alpha, beta and theta are saved
            path_save (string)           : path where to save the trained parameters

        Returns
        -------
            w(tensor)                     : reconstructed image in the wavelet domain, shape b*h*w*1

        """
        for i in range(self.nb_unrolledBlocks):
            w = self.Layers[i](w, wave_bp, rescal, self.RadonWave_transf, save_params, path_save)
        return w

            
            
        