import os
import numpy as np
from skimage.measure import label
import logging
import fpfs
import galsim
# import score.cadmos_lib as cl

def PSNR(img1, img2, normalize=False):
    """Calculate the PSNR of two images."""
    if not img1.shape == img2.shape:
        logging.raiseExceptions('Images have inconsistent Shapes!')

    img1 = np.array(img1)
    img2 = np.array(img2)

    if normalize:
        img1 = (img1 - img1.min())/(img1.max() - img1.min())
        img2 = (img2 - img2.min())/(img2.max() - img1.min())
        pixel_max = 1.0
    else:
        pixel_max = np.max([img1.max(), img2.max()])

    mse = ((img1 - img2)**2).mean()
    psnr = 20*np.log10(pixel_max/np.sqrt(mse))

    return psnr


def estimate_shear(obs, psf_in=None, use_psf=True, beta=0.5):
    """Estimate shear from input 2D image using fpfs 2.0.5."""
    
    # psf = np.zeros(obs.shape)
    # if use_psf: # Crop out PSF
    #     # beg = psf.shape[0]//2 - rcut
    #     # end = beg + 2*rcut + 1
    #     # psf = psf[beg:end,beg:end]
    #     # psf_pad = np.zeros((obs.shape[0], obs.shape[1]))
    #     starti = (obs.shape[0] - psf_in.shape[0]) // 2
    #     endi = starti + psf_in.shape[0]
    #     startj = (obs.shape[1] - psf_in.shape[1]) // 2
    #     endj = startj + psf_in.shape[1]
    #     psf[starti:endi, startj:endj] = psf_in
    # else: # Use delta for PSF if not given, equivalent to no deconvolution
    #     psf[obs.shape[0]//2, obs.shape[1]//2] = 1
    
    if psf_in is None:
        cen = ((np.array(obs.shape)-1.0)/2.0).astype(int)
        psf = np.zeros(obs.shape)
        psf[cen[0], cen[1]] = 1.0
    else:
        psf = psf_in
        
    fpTask = fpfs.fpfsBase.fpfsTask(psf, beta=beta)
    modes = fpTask.measure(obs)
    ells = fpfs.fpfsBase.fpfsM2E(modes, const=1, noirev=False)
    resp = ells['fpfs_RE'][0]
    g_1 = ells['fpfs_e1'][0] / resp
    g_2 = ells['fpfs_e2'][0] / resp
    g = np.sqrt(g_1 ** 2 + g_2 ** 2)
    
    # g = min(g, 1)

    return (g_1, g_2, g) 

def estimate_shear_new(obs, psf_in=None, use_psf=True, sigma_arcsec=0.6):
    """Estimate shear from input 2D image using fpfs 2.0.5."""
    
    if psf_in is None:
        cen = ((np.array(obs.shape)-1.0)/2.0).astype(int)
        psf = np.zeros(obs.shape)
        psf[cen[0], cen[1]] = 1.0
    else:
        psf = psf_in
    
    fpTask =  fpfs.image.measure_source(psf, noiFit=None, sigma_arcsec=sigma_arcsec, pix_scale=0.2)
    mms = fpTask.measure(obs-obs.min())
    ells = fpfs.catalog.fpfsM2E(mms,const=1,noirev=False)
    resp = ells['fpfs_R1E'][0]
    g_1 = ells['fpfs_e1'][0] / resp
    g_2 = ells['fpfs_e2'][0] / resp
    g = np.sqrt(g_1 ** 2 + g_2 ** 2)

    return (g_1, g_2, g) 

######################################################

def estimate_elli(obs):
    bg_level = get_background(obs)
    print(bg_level)
    n_row, n_col = obs.shape
    U = makeUi(n_row, n_col)
    GX = np.array([scal(obs-bg_level, U_i) for U_i in U])
    mu20 = 0.5*(GX[3] + GX[4]) - GX[0]**2 / GX[2]
    mu02 = 0.5*(GX[3] - GX[4]) - GX[1]**2 / GX[2]
    mu11 = GX[5] - GX[0] * GX[1] / GX[2]
    e1 = (mu20-mu02) / (mu20+mu02)
    e2 = 2*(mu11) / (mu20+mu02)
    e = np.sqrt(e1**2 + e2**2)
    # shear = galsim.Shear(e1=e1, e2=e2)
    
    # return (shear.g1, shear.g2, shear.g)
    return (e1, e2, e)

def get_background(img):
    mask = np.zeros_like(img)
    mask[0,:] = 1
    mask[-1,:] = 1
    mask[:,0] = 1
    mask[:,-1] = 1
    
    mask /= mask.sum()
    
    return (img * mask).sum()

#####################################

def makeU1(n,m):
    """This function returns a n x m numpy array with (i)_{i,j} entries where i
    is the ith line and j is the jth column.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U1 = np.tile(np.arange(n),(m,1)).T
    return U1

def makeU2(n,m):
    """This function returns a n x m numpy array with (j)_{i,j} entries where i
    is the ith line and j is the jth column.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U2 = np.tile(np.arange(m),(n,1))
    return U2

def makeU3(n,m):
    """This function returns a n x m numpy array with (1)_{i,j} entries where i
    is the ith line and j is the jth column.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U3 = np.ones((n,m))
    return U3

def makeU4(n,m):
    """This function returns a n x m numpy array with (i^2+j^2)_{i,j} entries 
    where i is the ith line and j is the jth column.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U4 = np.add.outer(np.arange(n)**2,np.arange(m)**2)
    return U4

def makeU5(n,m):
    """This function returns a n x m numpy array with (i^2-j^2)_{i,j} entries 
    where i is the ith line and j is the jth column.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U5 = np.subtract.outer(np.arange(n)**2,np.arange(m)**2)
    return U5

def makeU6(n,m):
    """This function returns a n x m numpy array with (i*j)_{i,j} entries where
    i is the ith line and j is the jth column.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U6 = np.outer(np.arange(n),np.arange(m))
    return U6

def makeUi(n,m):
    """This function returns a 6 x n x m numpy array containing U1, U2, U3, U4,
    U5 and U6.
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: 6 x n x m numpy array"""
    U1 = makeU1(n,m)
    U2 = makeU2(n,m)
    U3 = makeU3(n,m)
    U4 = makeU4(n,m)
    U5 = makeU5(n,m)
    U6 = makeU6(n,m)
    return np.array([U1,U2,U3,U4,U5,U6])

def scal(a,b):
    """This function returns the scalar product of a and b.
    INPUT: a, numpy array
           b, numpy array
    OUTPUT: scalar"""
    return (a*np.conjugate(b)).sum()


def bordering_blobs_mask(img):
    """This function keeps the biggest blob in the image considering the 
    gradient of the image.
    INPUT: img, Numpy Array
    OUTPUT: mask, boolean Numpy Array"""
    grad = np.abs(img-np.roll(img,1))
    threshold = np.quantile(grad,0.8)
    binary_grad = grad>threshold
    mask = blob_mask(binary_grad)
    return mask

def blob_mask(img,background=0,connectivity=2):
    """This function keeps the biggest blob in the image.
    INPUT: img, Numpy Array
           background, integer
           connectivity, integer
    OUTPUT: mask, boolean Numpy Array"""
    labels = label(img,background=background,connectivity=connectivity)
    #find the biggest blob
    indices = np.unique(labels)
    sizes = np.zeros(indices.shape)
    for i in indices[1:]:
        sizes[i] = (labels==i).sum()
    main_blob_label = np.argmax(sizes)
    main_blob_estimate = (labels==main_blob_label)*main_blob_label
    #extract mask
    mask = (labels-main_blob_estimate)==0
    return mask