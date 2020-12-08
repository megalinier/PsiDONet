# PsiDONet

### **Convolutional neural network for limited-angle tomographic reconstruction inspired by pseudodifferential operators**

- **License:** GNU General Public License v3.0
- **Author:**  Mathilde Galinier
- **Institution:** Università degli studi di Modena e Reggio Emilia
- **Doctoral programme:** INdAM Doctoral Programme in Mathematics and Applications Cofunded by Marie Sklodowska-Curie Actions (INdAM-DP-COFUND-2015) 
- **Email:** megalinier@gmail.com
- **Related publication:** https://arxiv.org/abs/2006.01620
- Please use the following citation:
  > T. A. Bubba, M. Galinier, M. Lassas, M. Prato, L. Ratti, and S. Siltanen.  '**Deep neural networks for inverse problems with pseudodifferential operators:  an ap-plication to limited-angle tomography**'. To appear in 
SIAM Journal on Imaging Sciences (SIIMS), 2020.

### Installation (Tensorflow environment)
1. Install anaconda.
2. Create an environment with python version 3.6.9.
```bash
$ conda create -n PsiDONet_tf_env python=3.6.9
```
3. Activate the environment
```bash
$ conda activate PsiDONet_tf_env 
```
4. Inside this environment install the following packages.
```bash
$ conda install -c anaconda numpy=1.14.6 
$ conda install -c conda-forge matplotlib=3.2.1 scikit-image=0.17.2 pywavelets=1.1.1
$ conda install tensorflow-gpu=1.13.1 
$ conda install -c astra-toolbox astra-toolbox=1.8.3
$ conda install -c odlgroup odl=0.7.0
```
5. Use the demo notebook to train and test PsiDONet models.

### Installation (PyTorch environment)
1. Install anaconda.
2. Create an environment with python version 3.6.9.
```bash
$ conda create -n PsiDONet_torch_env python=3.6.9
```
3. Activate the environment
```bash
$ conda activate PsiDONet_torch_env 
```
4. Inside this environment install the following packages.
```bash
$ conda install -c anaconda numpy=1.14.6 
$ conda install -c conda-forge matplotlib=3.2.1 scikit-image=0.17.2 pywavelets=1.1.1
$ conda install pytorch=1.0.0 torchvision=0.2.1 cuda80 -c pytorch
$ conda install -c astra-toolbox astra-toolbox=1.8.3
$ conda install -c odlgroup odl=0.7.0
```
5. Add the pytorch wavelet package with pip.
```bash
$ git clone https://github.com/fbcotter/pytorch_wavelets
$ cd pytorch_wavelets
$ pip install .
```
6. Use the demo notebook to train and test PsiDONet models.

### File organisation:

- ```Tensorflow```: Contains the tensorflow version of PsiDONet files
  - ```fundamental_functions```: Contains PsiDONet files
    - ```Train_Test_PsiDONet.py```: Contains the training and testing functions of PsiDONet
    - ```utils_learning.py```: Includes the definition of the PsiDONet unrolled iterations
    - ```utils_bowtie.py```: Contains functions for filter computation
    - ```tools.py```: Includes loading and quality evalutation functions
    - ```auxiliary_functions.py```: Contains some useful side functions
    - ```tf_wavelets.py```: Contains wavelet transform functions
    - ```haar_psi.py```: Contains the implementation of HaarPSI (code from [http://www.haarpsi.org/](http://www.haarpsi.org/)) 
  - ```demo.ipynb```: shows how to train and test PsiDONet with the tensorflow implementation

- ```PyTorch```: Contains the PyTorch version of PsiDONet files (only PsiDONetO and PsiDONetO+ are implemented)
  - ```fundamental_functions```: Contains PsiDONet files
    - ```Train_Test_PsiDONet.py```: Contains the training and testing functions of PsiDONet
    - ```model.py```: Includes the definition of the layers in PsiDONet
    - ```modules.py```: Contains useful functions employed in PsiDONet
    - ```tools.py```: Includes loading and quality evalutation functions
    - ```auxiliary_functions.py```: Contains some useful side functions
    - ```haar_psi.py```: Contains the implementation of HaarPSI 
  - ```PyTorch_ssim```: forward and backward functions to use SSIM as the training loss, (code from [https://github.com/Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)) 
  - ```demo.ipynb```: shows how to train and test PsiDONet with the PyTorch implementation

- ```Ellipses_Datasets```: Datasets of ellipse images
  - ```Size_128```: Dataset of 128x128 images
    - ```train```: Training set
      - ```Images```: Contains 10000 ground truth ellipse images
      - ```Sinograms```: Contains 10000 complete-angle sinograms
    - ```val```: Validation set
      - ```Images```: Contains 500 ground truth ellipse images
      - ```Sinograms```: Contains 500 complete-angle sinograms
    - ```test```: Test set
      - ```Images```: Contains 500 ground truth ellipse images
      - ```Sinograms```: Contains 500 complete-angle sinograms

The ground truth images and the sinograms were simulated with Matlab. The sinograms were corrupted by gaussian noise and generated according to a procedure that avoids inverse crime.

###### Thank you for using our code, kindly report any suggestion to the corresponding author.
