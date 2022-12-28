#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mar 30, 2015

@authors: mjiang, fnammour
'''
import numpy as np
from scipy.signal import convolve2d

def b3spline_fast(step_hole):
    """This function returns 2D B3-spline kernel for the 'a trou' algorithm.
    INPUT:  step_hole, non-negative integer(number of holes)
    OUTPUT: 2D numpy array """
    step_hole = int(step_hole)
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    length = 4*step_hole+1
    kernel1d = np.zeros((1,length))
    kernel1d[0,0] = c1
    kernel1d[0,-1] = c1
    kernel1d[0,step_hole] = c2
    kernel1d[0,-1-step_hole] = c2
    kernel1d[0,2*step_hole] = c3
    kernel2d = np.dot(kernel1d.T,kernel1d)
    return kernel2d

def star2d(im,scale,gen2=True):
    """This function returns the starlet transform of an image.
    INPUT:  im, 2D numpy array
            scale, positive integer (number of scales)
            gen2, boolean (to select the starlets generation)
    OUTPUT: 3D numpy array """
    (nx,ny) = np.shape(im)
    nz = scale
    wt = np.zeros((nz,nx,ny))
    step_hole = 1
    im_in = np.copy(im)
    
    for i in np.arange(nz-1):
        kernel2d = b3spline_fast(step_hole)
        im_out = convolve2d(im_in, kernel2d, boundary='symm',mode='same')
            
        if gen2:
            im_aux = convolve2d(im_out, kernel2d, boundary='symm',mode='same')
            wt[i,:,:] = im_in - im_aux
        else:        
            wt[i,:,:] = im_in - im_out
            
        im_in = np.copy(im_out)
        step_hole *= 2
        
    wt[nz-1,:,:] = np.copy(im_out)
    
    return wt
   
def istar2d(wtOri,gen2=True):
    """This function reconstructs the image from its starlet transformation.
    INPUT:  wtOri, 3D numpy array
            gen2, boolean (to precise the starlets generation)
    OUTPUT: 3D numpy array """
    (nz,nx,ny) = np.shape(wtOri)
    wt = np.copy(wtOri)
    if gen2:
        '''
        h' = h, g' = Dirac
        '''
        step_hole = pow(2,nz-2)
        imRec = np.copy(wt[nz-1,:,:])
        for k in np.arange(nz-2,-1,-1):
            kernel2d = b3spline_fast(step_hole)
            im_out = convolve2d(imRec, kernel2d, boundary='symm',mode='same')
            imRec = im_out + wt[k,:,:]
            step_hole /= 2            
    else:
        '''
        h' = h, g' = Dirac + h
        '''
        imRec = np.copy(wt[nz-1,:,:])
        step_hole = pow(2,nz-2)
        for k in np.arange(nz-2,-1,-1):
            kernel2d = b3spline_fast(step_hole)
            imRec = convolve2d(imRec, kernel2d, boundary='symm',mode='same')
            im_out = convolve2d(wt[k,:,:], kernel2d, boundary='symm',mode='same')
            imRec += wt[k,:,:]+im_out
            step_hole /= 2
    return imRec