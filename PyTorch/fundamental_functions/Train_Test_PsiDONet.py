# -*- coding: utf-8 -*-
"""
PSIDONET. Train and test.
PyTorch version.
                          
@author: Mathilde Galinier
@date: 07/12/2020
"""
import numpy as np
import os
import gc
import sys
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fundamental_functions.model import myModel, SSIM_loss
from fundamental_functions.tools import compute_angles, torch_dtype, Get_Rescal_Factor, get_nbDetectorPixels
from fundamental_functions.modules import MyDataset, OpenMat_transf, Reorg_wave_transf, Scikit_transf, \
                                          Missing_angle_sino, Reorg_wave_inverse_transf, Astra_transf, \
                                          compute_PSNR_SSIM
from pytorch_wavelets import DWT, IDWT
from torchvision import transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt

class PsiDONet_class(nn.Module):    
    """
    Includes the main training and testing methods of PsiDONet model.
    Attributes
        -----------
        missing_angle (int)         : missing angle on the first quadrant
        step_angle  (int)           : number of degrees between two consecutive 
                                      projections (>=1). 
        size_image (int)            : image dimension (multiple of 2)
        mu (float)                  : standard ISTA regularisation parameter
        L (float)                   : standard ISTA constant
        angles(numpy array)         : array of floats consisting of the angles of each measured projection 
        path_main (string)          : path to main folder containing all the 
                                      subfolders to be used
        path_datasets(string)       : path to the dataset
        path_save (string)          : path where to save the results
        path_train (string)         : path to the training dataset
        path_val (string)           : path to the validation dataset
        path_test (string)          : path to the test dataset
        nbDetectorPixels(int)       : width of a sinogram 
        mode(string)                :  'train' or 'test'                                
        model_unrolling (string)    : name of the version of PsiDONet used to train/test the model
                                      ('PSIDONetO' or 'PSIDONetOplus')
        learning_rate (float)       : learning rate used for the training
        nb_epochs (int)             : number of epochs used for the training
        minibatch_size (int)        : size of the minibatch used for the training
        loss_type (string)          : kind of cost function ('SSIM' or 'MSE')
        loss_domain (string)        : domain in which the evaluation of the cost function takes place ('IM' or 'WAV')
        loss_fun(method)            : computes the error between two images 
        nb_unrolledBlocks (int)     : number of different sets of trainable parameters 
        nb_repetBlock (int)         : number of times each set of trainable parameters is used 
        filter_size (int)           : size of each trainable filter
        wavelet_type (string)       : type of wavelets
        level_decomp (int)          : number of wavelet scales (J-J0-1 in the article)
        precision_float (int)       : machine precision (16 for half-precision, 
                                      32 for single-precision, 64 for double-precision)
        size_val_limit(int)         : size of the subset taken from the validation set, used to evaluate 
                                      the performance of the trained model during its training (for early stopping)
        modulo_batch(int)           : number of minibatches after which the model is evaluated on the validation set                             
        precision_type(torch.dtype) : torch type corresponding to the desired machine precision 
        dtype (torch.tensortype)    : cuda type corresponding to the desired machine precision      
        rescal_WRstarRWstar(float)  : rescaling factor equal to the inverse of an approximation of the norm of RstarR         
        model(myModel)              : PsiDONet layers
        train_loader(DataLoader)    : loader for the training set
        val_loader(DataLoader)      : loader for the validation set
        test_loader(DataLoader)     : loader for the test set
        size_train(int)             : number of images in the training set
        size_val(int)               : number of images in the validation set  
        size_test(int)              : number of images in the test set  
        
    """
    def __init__(self, train_conditions, folders, mode='train', model_unrolling ='PSIDONetO',
                learning_rate= 0.001, nb_epochs=3, minibatch_size=15, loss_type='MSE', loss_domain = 'WAV',
                nb_unrolledBlocks=40, nb_repetBlock=3, filter_size = 64,
                wavelet_type = 'haar', level_decomp = 3 , precision_float = 32, size_val_limit = 60,
                dataset = 'Ellipses'):
        """
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
            mode(string)              :  'train' or 'test'                                
            model_unrolling (string)  : name of the version of PsiDONet used to train/test the model
                                        ('PSIDONetO', 'PSIDONetOplus', 'PSIDONetF' or 'PSIDONetFplus')
            learning_rate (float)     : learning rate used for the training
            nb_epochs (int)           : number of epochs used for the training
            minibatch_size (int)      : size of the minibatch used for the training
            loss_type (string)        : kind of cost function ('SSIM' or 'MSE')
            loss_domain (string)      : domain in which the evaluation of the cost function takes place ('IM' or 'WAV')
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
        """

        super(PsiDONet_class, self).__init__()  
        
        #Checking values
        if filter_size%2==0:
            raise Exception('filter size must be odd.')  
            
        self.missing_angle, self.step_angle, self.size_image, self.mu, self.L = train_conditions
        self.angles            = compute_angles(self.missing_angle, self.step_angle)
        self.path_main, self.path_datasets, self.path_save                    = folders
        self.path_train        = os.path.join(self.path_datasets,'train')
        self.path_val          = os.path.join(self.path_datasets,'val')
        self.path_test         = os.path.join(self.path_datasets,'test')
        self.nbDetectorPixels  = get_nbDetectorPixels(self.path_datasets)
        
        self.mode              = mode
        self.model_unrolling   = model_unrolling
        self.learning_rate     = learning_rate
        self.nb_epochs         = nb_epochs
        self.minibatch_size    = minibatch_size
        self.loss_type         = loss_type
        self.loss_domain       = loss_domain
        self.nb_unrolledBlocks = nb_unrolledBlocks
        self.nb_repetBlock     = nb_repetBlock
        self.filter_size       = filter_size
        self.wavelet_type      = wavelet_type
        self.level_decomp      = level_decomp
        self.precision_float   = precision_float
        self.size_val_limit    = size_val_limit
        self.modulo_batch      = 25 #the stats of the training process are computed each modulo_batch minibatch
          
        if self.mode!='test':
            #definition of the loss function (à compléter)
            if self.loss_type=='SSIM':
                self.loss_fun  = SSIM_loss() 
            elif self.loss_type=='MSE':
                self.loss_fun  = torch.nn.MSELoss(reduction ='mean')
                
        self.precision_type, self.dtype = torch_dtype(self.precision_float)
        
        # Setting torch default type
        torch.set_default_dtype(self.precision_type) 
        
        # Wavelets Operations
        Wave_transf      = DWT(J=self.level_decomp, mode='periodization', wave=self.wavelet_type).cuda()
        Reorg_wave       = Reorg_wave_transf(self.level_decomp, self.dtype)         
        self.GotoWave    = transforms.Compose([Wave_transf,Reorg_wave])    
        Reorg_inv        = Reorg_wave_inverse_transf(self.level_decomp, self.dtype)
        Wstar_transf     = IDWT(mode='periodization', wave=self.wavelet_type).cuda()
        self.GoBackIm    = transforms.Compose([Reorg_inv, Wstar_transf])      
        
        # WR*RW*
        class_BP                 = Astra_transf(self.angles, self.size_image, self.nbDetectorPixels, self.precision_float)
        RstarR_transf            = class_BP.RstarR_transf     
        self.WRstarRWstar        = transforms.Compose([self.GoBackIm, RstarR_transf, self.GotoWave])
  
        self.rescal_WRstarRWstar = Get_Rescal_Factor(dim_inputs = [1, 1, self.size_image, self.size_image],\
                                         transf=RstarR_transf, type_input='tensor') 
                    
        self.model               = myModel(self.WRstarRWstar, self.nb_unrolledBlocks, nb_repetBlock, self.mu, self.L,\
                                         self.level_decomp, self.filter_size, self.model_unrolling,\
                                         self.size_image, self.dtype).cuda()
        
    def LoadData(self, return_wavelet = False):
        """
        According to the mode, creates the appropriate loader for the training, validation and test sets.
        """
        #Defining the functions required by MyDataset to open and transforms the images and sinograms computed with Matlab
        Missing_angle_transf = Missing_angle_sino(self.angles)
        Open_transf          = OpenMat_transf(self.dtype, permute_bool = 0)
        Scikit_class         = Scikit_transf(self.angles, self.size_image)
        BP_transf            = Scikit_class.Rstar_transf_shift
        Open_BP              = transforms.Compose([Missing_angle_transf, BP_transf, Open_transf]) 

        #Rescaling factor computed on the normal operator, in order to take into account also the rescaling factor of Matlab
        rescal_scikit        = Get_Rescal_Factor(dim_inputs = [self.size_image,self.size_image],\
                                                 transf=Scikit_class.RstarR_transf, type_input='numpy') 
            
        if self.mode =='train':
            train_data        = MyDataset(folder=self.path_train, Open_im=Open_transf, Open_BP = Open_BP,\
                                    rescaling_factor = rescal_scikit, return_wavelet = True, Open_Wave = self.GotoWave)
            self.train_loader = DataLoader(train_data, batch_size=self.minibatch_size, shuffle=True)
            self.size_train   = len([n for n in os.listdir(os.path.join(self.path_train,'Images'))])
        elif self.mode =='test':
            test_data         = MyDataset(folder=self.path_test, Open_im=Open_transf, Open_BP = Open_BP,\
                                    rescaling_factor = rescal_scikit, return_wavelet = True, Open_Wave = self.GotoWave)
            self.test_loader  = DataLoader(test_data, batch_size=self.minibatch_size, shuffle=False)
            self.size_test  = len([n for n in os.listdir(os.path.join(self.path_test,'Images'))])
            
        #Validation set
        val_data          = MyDataset(folder=self.path_val, Open_im=Open_transf, Open_BP = Open_BP,\
                                rescaling_factor = rescal_scikit, return_wavelet = True, Open_Wave = self.GotoWave)
        self.val_loader   = DataLoader(val_data, batch_size=self.minibatch_size, shuffle=False)
        self.size_val     = len([n for n in os.listdir(os.path.join(self.path_val,'Images'))])            
 
    def CreateFolders(self):
        """
        Creates directories for saving results.
        """

        if self.mode=='train':
            if not os.path.exists(self.path_save):
                os.makedirs(self.path_save)     
             
            subfolders = ['models','parameters','images','training_stats']
            paths         = [os.path.join(self.path_save, sub) for sub in subfolders ]
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
                    
        elif self.mode=='test':
            folder = os.path.join(self.path_save,'Testset_restoredImages')
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def PrintStatistics(self, val, epoch, i, loss, lr):
        """
        Prints information about the training.
        Parameters
        ----------
        val   (list): size 2*2, average PSNR and SSIM on the validation set and on the reconstructed validation images
        epoch  (int): epoch number
        i      (int): minibatch number
        loss (float): value of the training loss function
        lr   (float): learning rate
        """
        print('-----------------------------------------------------------------')
        print('[%d]'%(epoch),'[%d]'%(i),'average', self.loss_type,': %5.5f'%(loss), 'lr %.2E'%(lr))
        print('     Validation set:') 
        print('         PSNR blurred = %2.3f, PSNR pred = %2.3f'%(val[0,0],val[0,1]))
        print('         SSIM blurred = %2.3f, SSIM pred = %2.3f'%(val[1,0],val[1,1]))
    
    def SaveLoss_PSNR_SSIM(self, epoch, loss_epochs, psnr_ssim_val):
        """
        Plots and saves training results.
        Parameters
        ----------
        epoch            (int): epoch number
        loss_epochs     (list): value of the loss function at each epoch
        psnr_ssim_val   (list): average PSNR and SSIM on the back projection validation set and on the reconstructed validation images, size 2*2*epoch 
        """
        # plot and save loss for all epochs
        fig,(ax_loss) = plt.subplots(1,1,figsize=(4, 4))
        ax_loss.plot(loss_epochs[0:epoch+1])
        ax_loss.set_title('Minimal loss\n'+ "%5.2f" % np.min(loss_epochs))
        fig_name = os.path.join(self.path_save,'training_stats',"loss.png")
        plt.savefig(fig_name)
        plt.close(fig)
        # plot and save PSNR on validation set for all epochs
        self.MyPlot('Max PSNR', psnr_ssim_val[0,:,0:epoch+1],'psnr_validation_set.png')
        # plot and save SSIM on validation set for all epochs
        self.MyPlot('Max SSIM', psnr_ssim_val[1,:,0:epoch+1],'ssim_validation_set.png')
        print('Plots of the training loss, PSNR and SSIM during training are saved.')     
                        
    def MyPlot(self, title, vec, name):
        """
        Plots and save the SSIM or PSNR during training before and after deblurring.
        Parameters
        ----------
        title  (str): figure title
        vec   (list): average PSNR or SSIM on the validation set and on the reconstructed images, size 2*epoch
        name   (str): figure name
        """
        fig, ax = plt.figure(), plt.subplot(111)
        ax.plot(vec[0,:],'b',label='Blurred')
        ax.plot(vec[1,:],'g',label='Restored')
        ax.set_title(title + "%3.3f" %np.max(vec[1,:]))
        fig_name        = os.path.join(self.path_save,'training_stats',name)
        handles, labels = ax.get_legend_handles_labels()
        lgd             = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        fig.savefig(fig_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
    
                    
    def train(self):
        """
        Train \PsiDONet-O or \PsiDONet-Oplus
        """
        print('=================== Training model ===================')
        # Fixing seeds
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        np.random.seed(123)
        
        # Create Folder to save results
        self.CreateFolders()
  
        # Loads Data with images and wavelets
        self.LoadData(return_wavelet= True)
        
        # To store results
        loss_epochs       =  np.zeros(self.nb_epochs)
        psnr_ssim_val     =  np.zeros((2,2,self.nb_epochs*self.size_train//self.modulo_batch))
        loss_min_val      =  float('Inf')                   
        
        # Defines the optimizer
        lr        = self.learning_rate
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
     
        #save model at each epoch
        torch.save(self.model.state_dict(),os.path.join(self.path_save,'models','non_trained_model.pt'))        
        # ==========================================================================================================
        counter = 0
        # Trains for several epochs
        for epoch in range(0,self.nb_epochs): 
            gc.collect()
            
            # modifies learning rate
            if epoch>0: #multiplies the learning by 0.8 at each epoch
                lr        = lr*0.8 
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=lr)
                
            # goes  through all minibatches
            for i,minibatch in enumerate(self.train_loader,0):
                # sets training mode
                self.model.train() #sets the training mode
            
                [names, im_true, _ , wave_true, wave_bp] = minibatch    # get the minibatch
                im_true      = Variable(im_true.type(self.dtype),requires_grad=False)
                wave_true    = Variable(wave_true.type(self.dtype),requires_grad=False)
                wave_bp      = Variable(wave_bp.type(self.dtype),requires_grad=False)
                w0           = Variable(torch.zeros_like(wave_true),requires_grad=False)
                w_pred       = self.model(w0, wave_bp, rescal = self.rescal_WRstarRWstar,\
                                          save_params=True, path_save=self.path_save) 
                
                # Computes and prints loss
                if self.loss_domain   == 'WAV':
                    loss         = self.loss_fun(w_pred, wave_true)
                elif self.loss_domain == 'IM':
                    # Going back to domain image
                    im_pred      = self.GoBackIm(w_pred)
                    loss         = self.loss_fun(im_pred, im_true)
                    
                loss_epochs[epoch] += torch.Tensor.item(loss)
                sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                
                # sets the gradients to zero, performs a backward pass, and updates the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # saves images and compute validation set statistics
                if i%self.modulo_batch==0:  
                    counter += 1
                    with torch.no_grad(): #Disables gradients
                        #If we haven't go back to the image domain yet:
                        if self.loss_domain != 'IM':   
                            im_pred      = self.GoBackIm(w_pred)
                        # Save reconstructed images    
                        utils.save_image(im_pred.data,os.path.join(
                            self.path_save,'images',str(epoch)+'_'+str(i)+'_restored_images.png'),normalize=False)
        
                        # tests on validation set
                        self.model.eval()      # evaluation mode
                        psnr_ssim = np.zeros((2,2))
                        loss_current_val = 0
                        for j, minibatch_j in enumerate(self.val_loader,0):
                            print(counter)
                            print(j)
                            # To reduce the running time, we compute the statistics only on a subset of the validation set (counting size_val_limit elements)
                            if j >= self.size_val_limit//self.minibatch_size:
                                break
                            
                            [names, im_true, im_bp, wave_true, wave_bp] = minibatch_j    # get the minibatch
                            im_true      = Variable(im_true.type(self.dtype),requires_grad=False)
                            im_bp        = Variable(im_bp.type(self.dtype),requires_grad=False)
                            wave_true    = Variable(wave_true.type(self.dtype),requires_grad=False)
                            wave_bp      = Variable(wave_bp.type(self.dtype),requires_grad=False)
                            w0           = Variable(torch.zeros_like(wave_true),requires_grad=False)
                            w_pred       = self.model(w0, wave_bp, rescal = self.rescal_WRstarRWstar) 
                            
                            # Computes loss on validation set
                            im_pred      = self.GoBackIm(w_pred)
                            if self.loss_domain   == 'WAV':
                                loss_val     = self.loss_fun(w_pred, wave_true)
                            elif self.loss_domain == 'IM':
                                loss_val     = self.loss_fun(im_pred, im_true)                    
                            loss_current_val += torch.Tensor.item(loss_val)
                            
                            #for statistics
                            psnr_ssim += compute_PSNR_SSIM(im_true, im_bp, im_pred, self.size_val_limit) # compute PSNR an SSIM
                            
                        if loss_min_val>loss_current_val:
                            torch.save(self.model.state_dict(),os.path.join(self.path_save,'models','trained_model_MinLossOnVal.pt'))
                            loss_min_val = loss_current_val
                        psnr_ssim_val[:,:,counter] = psnr_ssim
                        # prints statistics
                        self.PrintStatistics(psnr_ssim_val[:,:,counter], epoch, i, loss_epochs[epoch],lr)
                        self.SaveLoss_PSNR_SSIM(counter, loss_epochs, psnr_ssim_val)
                               
            #save model at each epoch
            torch.save(self.model.state_dict(),os.path.join(self.path_save,'models','trained_model_ep'+str(epoch)+'.pt'))
        #==========================================================================================================
        # training is finished
        print('-----------------------------------------------------------------')
        print('Training is done.')
        print('-----------------------------------------------------------------')

    def test(self):  
        """
        Test the model.
        """
        print('=================== Testing model ===================')
        self.CreateFolders()
        
        for folder_to_test in ['test','val']:
            print('Saving restaured images in %s ...'%(os.path.join(self.path_save, folder_to_test + '_restoredImages')),flush=True)
            # Loads Data with images and wavelets
            self.LoadData(return_wavelet= True)
                
            with torch.no_grad(): #Disables gradients
                # evaluation mode
                self.model.eval() 
                for minibatch in self.test_loader:
                    [names, im_true, im_bp, wave_true, wave_bp] = minibatch    # get the minibatch
                    im_true      = Variable(im_true.type(self.dtype),requires_grad=False)
                    im_bp        = Variable(im_bp.type(self.dtype),requires_grad=False)
                    wave_true    = Variable(wave_true.type(self.dtype),requires_grad=False)
                    wave_bp      = Variable(wave_bp.type(self.dtype),requires_grad=False)
                    w0           = Variable(torch.zeros_like(wave_true),requires_grad=False)
                    w_pred       = self.model(w0, wave_bp, rescal = self.rescal_WRstarRWstar) 
                    
                    im_pred      = self.GoBackIm(w_pred)  
                    
                    # saves restored images
                    for j in range(len(names)):
                        im_predj = im_pred.data[j].squeeze(0).squeeze(1).cpu().numpy().astype('float'+str(self.precision_float))
                        im_predj[im_predj<0] = 0
                        sio.savemat(os.path.join(self.path_save, folder_to_test + '_restoredImages',names[j]+'.mat'),\
                                    {'image':im_predj})                   
