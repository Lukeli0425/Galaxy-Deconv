#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:04:42 2020

@author: fnammour
"""

from score import score
import numpy as np
import os

#define paths
root_path = '/Users/username/path/to/'
data_path = root_path+'data_folder/'

#load data
#set SNRs
SNRs = [40,75,150,380]

#Load psf
PSF = np.load(data_path+'psfs.npy')
gal_num,row,column = PSF.shape
digit_num = int(np.round(np.log10(gal_num)))+1

#set denoising parameters
n_starlet = 4 #number of starlet scales
n_shearlet = 3 #number of shearlet scales
lip_eps = 1e-3 #error upperbound for Lipschitz constant
tolerance = 1e-6 #to test convergence
n_itr = 150 #number of iteration
k = 4 #Set k for k-sigma hard thresholding
beta_factor = 0.95 #to ensure that beta is not too big
rip = False #Removal of Isolated Pixel in the deconvolution solution
first_guess = np.ones((row,column))/(row*column) #first guess
gamma = 0.0 #desactivate the shape constraint

#define result path
results_path = root_path+'results/{0}_conv_k{1}/'.format(n_itr,k)

#instantiate the solver
solver = score(k=k,n_starlet=n_starlet,n_shearlet=n_shearlet,epsilon=lip_eps,\
         rip=rip,tolerance=tolerance,beta_factor=beta_factor,gamma=gamma,\
         first_guess=first_guess,verbose=False)

#loop on SNR
for SNR in SNRs:
    #load observed galaxies
    obs_gals = np.load(data_path+'SNR{0}/noisy_galaxies_SNR{0}.npy'.format(SNR))
    obs_gals = obs_gals[:2]
    decon_list = list() #deconvolved galaxies list
    ell_list = list() #ellipticity list
    gal_counter = 1
    print('DECONVOLVING GAMMA={1} SNR={0}'.format(SNR,gamma))
    #loop on the galaxy images
    for Y,H in zip(obs_gals,PSF):
        solver.deconvolve(obs=Y,psf=H)
        decon_list += [solver.solution] 
        ell_list += [solver.ell_solution]
        nz = digit_num*'0'
        print(('Galaxy %'+nz+'{0}d of {1}').format(digit_num, gal_num) % gal_counter,end='\r')
        gal_counter += 1
        
    output_directory = results_path+'SNR{0}/'.format(SNR)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    np.save(output_directory+'deconvolved_galaxies_gamma_zero.npy',np.array(decon_list))
    np.save(output_directory+'ellipticities_gamma_zero.npy',np.array(ell_list))
    