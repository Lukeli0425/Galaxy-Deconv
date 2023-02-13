import numpy as np
from AlphaTransform import AlphaShearletTransform as AST
from numpy.linalg import norm
from scipy.signal import convolve


def makeU1(n,m):
    """Create a n x m numpy array with (i)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U1 = np.tile(np.arange(n),(m,1)).T
    return U1

def makeU2(n,m):
    """Create a n x m numpy array with (j)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U2 = np.tile(np.arange(m),(n,1))
    return U2

def makeU3(n,m):
    """Create a n x m numpy array with (1)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U3 = np.ones((n,m))
    return U3

def makeU4(n,m):
    """Create a n x m numpy array with (i^2+j^2)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U4 = np.add.outer(np.arange(n)**2,np.arange(m)**2)
    return U4

def makeU5(n,m):
    """Create a n x m numpy array with (i^2-j^2)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U5 = np.subtract.outer(np.arange(n)**2,np.arange(m)**2)
    return U5

def makeU6(n,m):
    """Create a n x m numpy array with (i*j)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U6 = np.outer(np.arange(n),np.arange(m))
    return U6

def makeUi(n,m):
    """Create a 6 x n x m numpy array containing U1, U2, U3, U4, U5 and U6
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: 6 x n x m numpy array"""
    U1 = makeU1(n,m)
    Ul = U1**2
    Uc = Ul.T
    return np.array([U1,U1.T,makeU3(n,m),Ul+Uc,Ul-Uc,makeU6(n,m)])

def make_Ui(n,m):
    return np.array([makeU1(n,m),makeU2(n,m),makeU3(n,m),makeU4(n,m),makeU5(n,m),makeU6(n,m)])

def get_shearlets(n_row, n_column, n_scale):
    """This function returns the normalized coefficients of the shearlets and 
    their adjoints.
    INPUT: n_row, positive integer
           n_column, positive integer
           n_scale, positive integer
    OUTPUT: shearlets, 3D numpy array
            adjoints, 3D numpy array"""
    #Get shearlet filters
    trafo = AST(n_column, n_row, [0.5]*n_scale, real=True, parseval=True, verbose=False, use_fftw=False)
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

def normalize(signal):
    """This function returns the normalized signal.
    INPUT: signal, numpy array of at least 2 dimensions
    OUTPUT: numpy array of the same shape than signal"""
    return np.array([s/norm(s) for s in signal])


def comp_mu(adj):
    n = adj.shape[-1]
    mu = np.array([[n/norm(im)**2 if not(np.isclose(norm(im),0)) else 0 for im in u]
                                                            for u in adj])
    return n*mu/mu.size


def get_adjoint_coeff(trafo):
    """Generates the coefficients of the adjoint operator of the shearlets"""
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



if __name__ == "__main__":
    import torch
    fov_pixels = 48
    n_shearlet = 2
    U = makeUi(fov_pixels, fov_pixels)
    shearlets, shearlets_adj = get_shearlets(fov_pixels, fov_pixels, n_shearlet)
    # shealret adjoint of U, i.e Psi^{Star}(U)
    psu = torch.Tensor(np.array([convolve_stack(ui, shearlets_adj) for ui in U]))
    mu = torch.Tensor(comp_mu(psu))
    
    print(psu.shape)
    print(mu.shape)