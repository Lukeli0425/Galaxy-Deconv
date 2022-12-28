#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:52:11 2019

@author: fnammour
"""

import numpy as np
from numpy.linalg import norm
from scipy.signal import convolve
from skimage.measure import label
from AlphaTransform import AlphaShearletTransform as AST

def rotate180(img):
    """This function rotates an image by 180 degrees.
    INPUT: img 2D numpy array (image)
    OUTPUT: n x m numpy array """
    return np.rot90(img, k=2, axes=(0, 1))

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

def get_adjoint_coeff(trafo):
    """This function returns the coefficients of the adjoint operator of the
    shearlets.
    INPUT: trafo, AlphaShearletTransform object
    OUTPUT: 3D numpy array"""
    column = trafo.width
    row = trafo.height
    n_scales = len(trafo.indices)
    #Attention: the type of the output of trafo.adjoint_transform is complex128
    #and by creating coeff without specifying the type it is set to float64
    #by default when using np.zeros
    coeff = np.zeros((n_scales,row,column))
    for s in range(n_scales):
        temp = np.zeros((n_scales,row,column))
        temp[s,row//2,column//2]=1
        coeff[s] = trafo.adjoint_transform(temp, do_norm=False)
    return coeff

def normalize(signal):
    """This function returns the normalized signal.
    INPUT: signal, numpy array of at least 2 dimensions
    OUTPUT: numpy array of the same shape than signal"""
    return np.array([s/norm(s) for s in signal])

def get_shearlets(n_row,n_column,n_scale):
    """This function returns the normalized coefficients of the shearlets and 
    their adjoints.
    INPUT: n_row, positive integer
           n_column, positive integer
           n_scale, positive integer
    OUTPUT: shearlets, 3D numpy array
            adjoints, 3D numpy array"""
    #Get shearlet filters
    trafo = AST(n_column, n_row, [0.5]*n_scale,real=True,parseval=True
                ,verbose=False)
    shearlets = trafo.shearlets
    adjoints = get_adjoint_coeff(trafo)
    #Normalize shearlets filter banks
    adjoints = normalize(adjoints)
    shearlets = normalize(shearlets)
    return shearlets,adjoints

def convolve_stack(img,kernels):
    """This function returns an array of the convolution result of img with
    each kernel of kernels.
    INPUT: img, 2D numpy array
           kernels, 3D numpy array
    OUTPUT: 3D numpy array"""
    return np.array([convolve(img,kernel,mode='same') for kernel in kernels])

def comp_mu(adj):
    """This function returns the weights mu of the shape constraint.
    INPUT: adj, 3D numpy array (The adjoint shearlet transform of U)
    OUTPUT: 1D numpy array"""
    n = adj.shape[-1]
    mu = np.array([[1/norm(im)**2 if not(np.isclose(norm(im),0)) else 0 for im in u]
                                                            for u in adj])
    return n*mu/mu.size

def scal(a,b):
    """This function returns the scalar product of a and b.
    INPUT: a, numpy array
           b, numpy array
    OUTPUT: scalar"""
    return (a*np.conjugate(b)).sum()

def comp_grad(R,adj_U,mu,gamma):
    """This function returns the gradient of the differentiable part of the
    loss function.
    INPUT: R, 2D numpy array (residual)
           adj_U, 3D numpy array (adjoint shearlet transform of U)
           mu, 1D numpy array (weights associated to adj_U)
           gamma, scalar (trade-off between data-fidelity and shape constraint)
    OUTPUT: 2D numpy array"""
    temp = gamma*np.array([[cst*scal(R,im)*im 
                            for cst,im in zip(m, u)]
                             for m,u in zip(mu,adj_U)]).sum((0,1)) + R
    return 2*temp

def eigenvalue(Op, v):
    """This function returns the scalar product of v and Op(v).
    INPUT: Op, function
           v, numpy array
    OUTPUT: scalar"""
    Op_v = Op(v)
    return scal(v,Op_v)

def power_iteration(Op, output_dim,epsilon=0.001):
    """This function returns the norm of the operator using the power iteration
    method.
    INPUT: Op, function
           output_dim, tuple (dimension of the operator 2D entry)
           epsilon, positive float (error upper bound)
    OUTPUT: scalar"""
    d = np.prod(output_dim)

    v = np.ones(d) / np.sqrt(d)
    v = v.reshape(output_dim)
    
    ev = eigenvalue(Op, v)

    while True:
        Op_v = Op(v)
        v_new = Op_v / np.linalg.norm(Op_v)

        ev_new = eigenvalue(Op, v_new)
        if np.abs(ev - ev_new) < epsilon:
            break

        v = v_new
        ev = ev_new
        
    return ev_new, v_new

def norm1(signal):
    """This function returns the l1-norm (for vecotrs) of a signal.
    INPUT: signal, Numpy Array
    OUTPUT: norm1_signal, scalar"""
    norm1_signal = norm(signal.flatten(),ord=1)
    return norm1_signal
    
def compute_background_mask(img,p=1,q=4,center=None):
    """This function returns a binary mask of an image where all the value are
    set to one except the square which center is given in input and size is 
    $\left(\frac{p}{q}\right)^2$ the size of the image.
    INPUT: img, Numpy Array
           p (optional), positive integer
           q (optional), positive integer
           center (optional), tuple of positive integers
    OUTPUT: norm1_signal, scalar"""
    n_lines,n_columns = img.shape
    x_slice,y_slice = p*n_lines//q,p*n_columns//q
    if (center == None).any():
        x_c,y_c = n_lines//2,n_columns//2
    else:
        x_c,y_c=center
    background_mask = np.ones(img.shape,dtype=bool)
    background_mask[x_c-x_slice:x_c+x_slice,y_c-y_slice:y_c+y_slice] = False
    return background_mask

def sigma_mad(signal):
    """This function returns the estimate of the standard deviation of White
    Additive Gaussian Noise using the Mean Absolute Deviation method (MAD).
    INPUT: signal, Numpy Array
    OUTPUT: sigma, scalar"""
    sigma = 1.4826*np.median(np.abs(signal-np.median(signal)))
    return sigma

def hard_thresh(signal, threshold):
    """This function returns the result of a hard thresholding operation.
    INPUT: signal, Numpy Array
           threshold, Numpy Array
    OUTPUT: res, Numpy Array"""
    res = signal*(np.abs(signal)>=threshold)
    return res

def MS_hard_thresh(wave_coef, n_sigma):
    """This function returns the result of a multi-scale hard thresholding
    operation perfromed on wave_coef and using the coefficients of n_sigma as
    thresholds.
    INPUT: wave_coef, Numpy Array
           n_sigma, Numpy Array
    OUTPUT: wave_coef_rec_MS, Numpy Array"""
    wave_coef_rec_MS = np.zeros(wave_coef.shape)
    for i,wave in enumerate(wave_coef):
        # Denoise image
        wave_coef_rec_MS[i,:,:] = hard_thresh(wave, n_sigma[i])
    return wave_coef_rec_MS

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