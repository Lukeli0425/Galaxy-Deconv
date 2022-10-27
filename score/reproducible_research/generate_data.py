#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:42:54 2019

@authors: fnammour and aguinot
"""

from astropy.io import fits
import galsim
import numpy as np
from score import score
import os

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
path_catalog = '/Users/fnammour/Documents/Librairies/Axel_code/generate_sersic/COSMOS_25.2_training_sample/'
data_path = '/Users/fnammour/Desktop/test/'
check_dir(data_path)

# Set numpy seed
np.random.seed(42)
g_seed = galsim.UniformDeviate(42)

d = fits.getdata(path_catalog+'real_galaxy_catalog_25.2_fits.fits', 1)

d.dtype.names

ind = np.where(d['viable_sersic'] == 1)

s_hlr = d['HLR'][:,0][ind] #In arcsec
s_n = d['sersicfit'][:,2][ind] #Sersic index
s_flux = d['FLUX'][:,0][ind]

#Select Sersic with hlr higher than hlr_min
hlr_min = 0.21
s_n = s_n[hlr_min<s_hlr]
s_flux = s_flux[hlr_min<s_hlr]
s_hlr = s_hlr[hlr_min<s_hlr]

#s_n values must be in [n_min,n_max] for galsim to work
n_min = 0.8
n_max = 2.3

s_hlr = s_hlr[n_min<s_n]
s_flux = s_flux[n_min<s_n]
s_n = s_n[n_min<s_n]

s_hlr = s_hlr[s_n<n_max]
s_flux = s_flux[s_n<n_max]
s_n = s_n[s_n<n_max]

def get_g(g_sigma, n_g):
    g = np.random.normal(0., g_sigma, n_g)
    while np.linalg.norm(g) > 1:
       g = np.random.normal(0., g_sigma, n_g)
    return g

gal_num = 300

#Give it an ellipticity by applying a shear to it
g_sigma = 0.3  # ref: Bernstein & Armstrong 2014

##Start by creating an isotropic PSF with a Moffat profile
#For a Moffat profile, we need to set the parameter beta
beta = 4.765 # Literature

#Full Width at Half Maximum (FWHM)
fwhm_min = 0.1
fwhm_max = 0.2

g_sigma_psf = 0.03
sig_noise = 0.1 #set the noise std

#Sampling (generate an image array out of the galsim object)
nx = 96 #set the number of columns
ny = 96 #set the number of lines
pixel_scale = 0.05

#Generate a score instance to estimate ground truth ellipticity
estimator = score()
estimator.n_row, estimator.n_col = nx,ny
estimator.set_defaults()
estimator.init_const()

SNRs = [40,75,150,380]

list_gal = []
list_ell = []
list_psf = []
list_gal_conv = []
list_ell_conv = []
list_gal_obs = [[]]*len(SNRs)

n_digit = int(np.ceil(np.log10(gal_num)))

for i in range(gal_num):
    #Generate a galaxy
    #Start by creating an isotropic galaxy
    print('{0:0>{1}d}/{2}'.format(i,n_digit,gal_num),end='\r')
    gal = galsim.Sersic(n=s_n[i], half_light_radius=s_hlr[i], flux=s_flux[i])
    
    e1,e2 = get_g(g_sigma, 2)
    gal = gal.shear(g1=e1, g2=e2)
    
    #ground truth galaxy image
    true_img = np.copy(gal.drawImage(nx=nx, ny=ny, scale=pixel_scale).array)
    list_gal += [true_img]
    list_ell += [estimator.estimate_ell(true_img)]
    
    #Generate a PSF
    fwhm = fwhm_min + np.random.rand() * (fwhm_max - fwhm_min)
    psf = galsim.Moffat(beta=beta, fwhm=fwhm)
    
    psf_e1 = get_g(g_sigma_psf, 1) #+cst1 because the mean of the PSF ellipticity
    psf_e2 = get_g(g_sigma_psf, 1) #+cst2 can be different from zero
    psf = psf.shear(g1=psf_e1, g2=psf_e2)
    
    #Convolve the galaxy with the PSF
    obj = galsim.Convolve(gal, psf)
    
    #Sampling (generate an image array out of the galsim object)
    galsim_img = obj.drawImage(nx=nx, ny=ny, scale=pixel_scale)
    psf_img = psf.drawImage(nx=nx, ny=ny, scale=pixel_scale)
    
    list_psf += [psf_img.array]
    conv_img = np.copy(galsim_img.array)
    list_gal_conv += [conv_img]
    list_ell_conv += [estimator.estimate_ell(conv_img)]
    
    for ind_snr, snr in enumerate(SNRs):
        #add Noise
        obs_img = galsim_img.copy()
        g_noise = galsim.GaussianNoise(sigma=sig_noise, rng=g_seed) #Generate Noise
        g_sig_noise = obs_img.addNoiseSNR(noise=g_noise, snr=snr, preserve_flux=True)
        list_gal_obs[ind_snr] = list_gal_obs[ind_snr]+[np.copy(obs_img.array)]

np.save(data_path+'galaxies.npy',np.array(list_gal))
np.save(data_path+'ellipticities.npy',np.array(list_ell))
np.save(data_path+'PSFs.npy',np.array(list_psf))
np.save(data_path+'convolved_galaxies.npy',np.array(list_gal_conv))
np.save(data_path+'convolved_ellipticities.npy',np.array(list_ell_conv))

array_gal_obs = np.array(list_gal_obs)

for ind_snr,snr in enumerate(SNRs):
    output_path = data_path+'SNR{}/'.format(snr)
    check_dir(output_path)
    np.save(output_path+'observed_galaxies_SNR{}.npy'.format(snr),array_gal_obs[ind_snr])
print('Done')