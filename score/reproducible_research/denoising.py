#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:04:42 2020

@author: fnammour
"""

from score import score
import numpy as np
import pickle
import os

#define paths
root_path = '/Users/username/path/to/'
data_path = root_path+'data_folder/'

#load data
#set SNRs and load the SNR-gamma correspondence dictionary
SNRs = [40,75,150,380]
pickle_in = open('./SNR_gamma_den.pkl','rb')
SNR_gamma = pickle.load(pickle_in)

#Load data set info
gal_num,row,column = 300,96,96
digit_num = int(np.round(np.log10(gal_num)))+1

#set denoising parameters
n_starlet = 4 #number of starlet scales
n_shearlet = 3 #number of shearlet scales
lip_eps = 1e-3 #error upperbound for Lipschitz constant
n_itr = 40 #number of iteration
k = 4 #Set k for k-sigma hard thresholding
beta_factor = 0.95 #to ensure that beta is not too big
rip = True #Removal of Isolated Pixel in the solution
first_guess = np.ones((row,column))/(row*column) #first guess

#define result path
results_path = root_path+'results/{0}_itr_k{1}/'.format(n_itr,k)

#solver with gamma = 0
g0 = score(k=k,n_starlet=n_starlet,n_shearlet=n_shearlet,epsilon=lip_eps,\
       rip=False,beta_factor=beta_factor,gamma=0.0,first_guess=first_guess,\
       verbose=False)

#loop on SNR
for SNR in SNRs:
    #set gamma according to the SNR
    gamma = SNR_gamma[SNR]
    #load observed galaxies
    obs_gals = np.load(data_path+'SNR{0}/noisy_galaxies_SNR{0}.npy'.format(SNR))
    obs_gals = obs_gals[:2]
    #instantiate the solver
    g_star = score(k=k,n_starlet=n_starlet,n_shearlet=n_shearlet,epsilon=lip_eps,\
               rip=rip,beta_factor=beta_factor,first_guess=first_guess,\
               gamma=gamma,verbose=False)
    g0_list = list() #denoised galaxies with gamma zero list
    g_star_list = list() #denoised galaxies with gamma star list
    ell0_list = list() #ellipticities with gamma zero list
    ell_star_list = list() #ellipticity with gamma star list
    gal_counter = 1
    print('DECONVOLVING GAMMA={1} SNR={0}'.format(SNR,gamma))
    #loop on the galaxy images
    for Y in obs_gals:
        g0.denoise(obs=Y)
        g0_list += [g0.solution]
        g_star.denoise(obs=Y,first_guess=g0.solution)
        g_star_list += [g_star.solution]
        nz = digit_num*'0'
        print(('Galaxy %'+nz+'{0}d of {1}').format(digit_num, gal_num) % gal_counter,end='\r')
        gal_counter += 1
        
    output_directory = results_path+'SNR{0}/'.format(SNR)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    np.save(output_directory+'denoised_galaxies_gamma_zero.npy',np.array(g0_list))
    np.save(output_directory+'denoised_galaxies_gamma_star.npy',np.array(g_star_list))
    np.save(output_directory+'ellipticities_gamma_zero.npy',np.array(ell0_list))
    np.save(output_directory+'ellipticities_gamma_star.npy',np.array(ell_star_list))