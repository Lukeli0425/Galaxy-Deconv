import numpy as np
from numpy.linalg import norm
# from modopt.signal.wavelet import filter_convolve
from scipy.signal import convolve
from skimage.measure import label
from utils.AlphaTransform import AlphaShearletTransform as AST


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

# def rotate180(img):
#     return np.rot90(img, k=2, axes=(0, 1))

# def hard_thresh(signal, threshold):
#     return signal*(np.abs(signal)>=threshold)

# def sigma_mad(signal):
#     return 1.4826*np.median(np.abs(signal-np.median(signal)))

# def MS_hard_thresh(wave_coef, n_sigma):
#     wave_coef_rec_MS = np.zeros(wave_coef.shape)
#     for i,wave in enumerate(wave_coef):
#         # Denoise image
#         wave_coef_rec_MS[i,:,:] = hard_thresh(wave, n_sigma[i])
#     return wave_coef_rec_MS

# def norm1(signal):
#     return np.abs(signal).sum()

# def shear_norm(signal,shearlets):
#     shearlet_norms = np.array([norm(s) for s in shearlets])
#     return np.array([s/n for s,n in zip(signal,shearlet_norms)])

# def scal(a,b):
#     return (a*np.conjugate(b)).sum()

# def G(X,U,W = 1):
#     return np.array([scal(X*W,U[i]) for i in range(6)])

# def prior(alpha):
#     norm = 0
#     for wave in alpha:
#         norm += norm1(wave)
#     return norm 

# def comp_adj(imgs,adjoints):
#     return np.array([filter_convolve(i,adjoints) for i in imgs])

def comp_mu(adj):
    n = adj.shape[-1]
    mu = np.array([[n/norm(im)**2 if not(np.isclose(norm(im),0)) else 0 for im in u]
                                                            for u in adj])
    return n*mu/mu.size

# def dirac(size):
#     D = np.zeros(size)
#     m,n = size
#     D[m//2,n//2]=1
#     return D

# def comp_kronecker_delta(size,position):
#     mat  = np.zeros(size)
#     mat[position] = 1
#     return mat

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

# def comp_grad(R,adj_U,mu,gamma):
#     temp = gamma*np.array([[cst*scal(R,im)*im 
#                             for cst,im in zip(m, u)]
#                              for m,u in zip(mu,adj_U)]).sum((0,1)) + R
#     return 2*temp

# def comp_thresh(alpha,k=4):
#     thresholds = []
#     thresholds += [(k+1)*sigma_mad(alpha[0])]
#     for wave in alpha[1:-1]:
#         thresholds += [k*sigma_mad(wave)]
#     return np.array(thresholds)

# def update_thresholds(R,filters,thresholds,k,itr,first_run):
#     R = R.reshape(R.shape[-2],R.shape[-1])
#     alphaR = filter_convolve(R, filters)
#     if first_run and itr < 5:
#         thresholds = comp_thresh(alphaR,k)    
#     return thresholds

# def comp_loss(R,alpha,gamma,mu,adj_U):
#     return np.array([norm(R)**2/2.,gamma*(np.array(
#             [[cst*((R*im).sum())**2*im for cst,im in zip(m, u)]
#             for m,u in zip(mu,adj_U)])/2.).sum(),prior(alpha)])

# def reconstruct(alpha, positivity = True):
#     X = alpha.sum(0)
#     if positivity:
#         X = X*(X>0)
#     return X

# def FindEll(X, U, W = 1):
#     GX = G(X,U,W)
#     mu20 = 0.5*(GX[3]+GX[4])-GX[0]**2/GX[2]
#     mu02 = 0.5*(GX[3]-GX[4])-GX[1]**2/GX[2]
#     mu11 = GX[5]-GX[0]*GX[1]/GX[2]
#     e1 = (mu20-mu02)/(mu20+mu02)
#     e2 = 2*(mu11)/(mu20+mu02)
#     return np.array([e1,e2])

# def comp_ell(gal,U):
#     return np.array([FindEll(g,U) for g in gal])

# def build_Qij(mu_ij,adj_ij):
#     qij = mu_ij*np.outer(adj_ij,adj_ij)
#     return qij

# def build_Q(mu,adj):
#     n = adj.shape[-1]
#     q = np.zeros((n,n))
#     for mu_i,adj_i in zip(mu,adj):
#         for mu_ij,adj_ij in zip(mu_i,adj_i):
#             q += build_Qij(mu_ij,adj_ij)
#     return q

# def build_S(gamma,Q):
#     n_row,n_col=Q.shape
#     return np.eye(n_row,n_col)+gamma*Q

# def build_C(h,S):
#     h_tilde = np.flip(h,0)
#     h1 = np.convolve(h,h_tilde,mode='same')
#     h1 = h1[:,None]
#     C = convolve(S,h1,mode='same')
#     return C

# def get_SC(gamma,mu,h,adj):
#     Q = build_Q(mu,adj)
#     S = build_S(gamma,Q)
#     C = build_C(h,S)
#     return S,C

# def flatten_list_im(list_im):
#     l_shape = np.array(list_im.shape)
#     f_shape = l_shape[:-1]
#     f_shape[-1] *= l_shape[-1]
#     return list_im.reshape(f_shape)

# def change_basis(basis,mat):
#     vec = mat.flatten()
#     new_vec = basis@vec
#     new_mat = new_vec.reshape(mat.shape)
#     return new_mat

# def eigenvalue(Op, v):
#     Op_v = Op(v)
#     return scal(v,Op_v)

# def power_iteration(Op, output_dim,epsilon=0.001):
#     d = np.prod(output_dim)

#     v = np.ones(d) / np.sqrt(d)
#     v = v.reshape(output_dim)
    
#     ev = eigenvalue(Op, v)

#     while True:
#         Op_v = Op(v)
#         v_new = Op_v / np.linalg.norm(Op_v)

#         ev_new = eigenvalue(Op, v_new)
#         if np.abs(ev - ev_new) < epsilon:
#             break

#         v = v_new
#         ev = ev_new
        
#     return ev_new, v_new

# def norm_op(Op):
#     op_shape = Op(1).shape
#     norm_OP = 0
#     for i in range(op_shape[0]):
#         for j in range(op_shape[1]):
#             arg = comp_kronecker_delta(op_shape,(i,j))
#             norm_OP += norm(Op(arg))**2
#     norm_OP = np.sqrt(norm_OP)
#     return norm_OP

# def std_op(Op):
#     op_shape = Op(1).shape
#     std_OP = []
#     for i in range(op_shape[0]):
#         std_OP_row = []
#         for j in range(op_shape[1]):
#             arg = comp_kronecker_delta(op_shape,(i,j))
#             cov = Op(Op(arg.T).T)
#             std_OP_row += [np.sqrt(cov[i,j])]
#         std_OP += [std_OP_row]      
#     return np.array(std_OP)

# def std_filters_op(Op,filters):
#     op_shape = Op(1).shape
#     std_filters_OP = []
#     for f in filters:
#         std_OP = []
#         for i in range(op_shape[0]):
#             std_OP_row = []
#             for j in range(op_shape[1]):
#                 arg = comp_kronecker_delta(op_shape,(i,j))
#                 temp = Op(arg.T)
#                 temp = filter_convolve(temp,f.reshape((1,f.shape[0],f.shape[1]))).reshape(temp.shape)
#                 cov = Op(temp.T)
#                 cov = filter_convolve(cov,f.reshape((1,f.shape[0],f.shape[1]))).reshape(temp.shape)
#                 std_OP_row += [np.sqrt(cov[i,j])]
#             std_OP += [std_OP_row]  
#         std_filters_OP += [std_OP]
#     return np.array(std_filters_OP)

# def std_3D_op(Op):
#     op_shape = Op(1).shape
#     std_OP = np.zeros(op_shape)
#     for i in range(op_shape[1]):
#         for j in range(op_shape[2]):
#             arg = comp_kronecker_delta(op_shape[1:],(i,j))
#             temp = np.swapaxes(Op(arg.T),1,2)
#             cov = np.array([Op(temp[i])[i] for i in range(op_shape[0])])
#             print(cov[:,i,j])
#             std_OP[:,i,j] = cov[:,i,j]
#     return np.array(std_OP)


# def gaussian_1D(x,mu,sigma):
#     exp_arg = -(x-mu)**2/2*sigma**2
#     mult_arg = 1.0/(np.sqrt(2*np.pi)*sigma)
#     return mult_arg*np.exp(exp_arg)

# def compute_background_mask(img,p=1,q=4,center=None):
#     n_lines,n_columns = img.shape
#     x_slice,y_slice = p*n_lines//q,p*n_columns//q
#     if center == None:
#         x_c,y_c = n_lines//2,n_columns//2
#     else:
#         x_c,y_c=center
#     background_mask = np.ones(img.shape,dtype=bool)
#     background_mask[x_c-x_slice:x_c+x_slice,y_c-y_slice:y_c+y_slice] = False
#     return background_mask

# def wavelet_adjoint(alpha,filters_tilde):
#     temp = np.array([convolve(a,w,'same') for a,w in zip(alpha,filters_tilde)])
#     return temp.sum(0)

# def blob_mask(img,background=0,connectivity=2):
#     '''Keep the biggest blob by remove the others.'''
#     labels = label(img,background=background,connectivity=connectivity)
#     #find the biggest blob
#     indices = np.unique(labels)
#     sizes = np.zeros(indices.shape)
#     for i in indices[1:]:
#         sizes[i] = (labels==i).sum()
#     main_blob_label = np.argmax(sizes)
#     main_blob_estimate = (labels==main_blob_label)*main_blob_label
#     #extract mask
#     mask = (labels-main_blob_estimate)==0
#     return mask



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