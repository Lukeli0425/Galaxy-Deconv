import os
import numpy as np
import logging
import fpfs


def PSNR(img1, img2, normalize=True):
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


def estimate_shear(obs, psf_in=False, use_psf=True, beta=0.5):
    """Estimate shear from input 2D image."""
    
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
    
    if not use_psf:
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
    
    g = min(g, 1)

    return (g_1, g_2, g) 



