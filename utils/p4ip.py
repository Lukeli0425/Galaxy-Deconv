import numpy as np
import torch 

import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift
from scipy.optimize import fmin_l_bfgs_b as l_bfgs
from torch.autograd import Variable
from utils_poisson_deblurring.utils_torch import conv_fft, img_to_tens, scalar_to_tens
from bm3d import bm3d, BM3DProfile


def l2_deconv(y, k_fft, lambda_reg):
	rhs = np.conj(k_fft)*fft2(y)
	lhs = np.abs(k_fft)**2 + lambda_reg
	return np.real(ifft2(rhs/lhs))

def psf_to_otf(ker, size):
	k_pad = np.zeros(size)
	centre = ker.shape[1]//2 + 1
	k_pad[:centre, :centre] = ker[(centre-1):, (centre-1):]
	k_pad[ :centre, -(centre-1):] = ker[ (centre-1):, :(centre-1)]
	k_pad[-(centre-1):, :centre] = ker[ : (centre-1), (centre-1):]
	k_pad[-(centre-1):, -(centre-1):] = ker[ :(centre-1), :(centre-1)]
	k_fft = fft2(k_pad)
	return k_pad, k_fft

def deblurring_3split(x0, x1, rho1, rho2, k_fft):
	num = fft2(x0 + (rho2/rho1)*x1)
	den = 1 + (rho2/rho1)*(np.abs(k_fft)**2)
	return np.real((ifft2(num/den)))

def poisson_proximal_3split(v0, y, M, rho2):
	v1 = (rho2*v0-M)
	return (v1+np.sqrt(v1**2 + 4*rho2*y))/(2*rho2)

def x_subp(x, params):
	eps = 1e-10
	a, b, c = -1/(2*eps**2), 2/eps, np.log(eps)-1.5

	y = params['y']
	H, W = np.shape(y); N = H*W
	y = np.reshape(y,[N])
	A = params['A']
	At = params['At']
	rho = params['rho']
	x0 = params['x0']
	Ax = A(np.reshape(x,[H,W])); Ax = np.reshape(Ax, [N])
	idx_zero = np.where( Ax < eps)
	idx_non_zero = np.where( Ax >= eps)
	du = np.zeros(np.shape(y), dtype=np.float32)
	if np.shape(idx_zero)[0] > 0:
		f_poly = -y[idx_zero]*(a*Ax[idx_zero]**2 + b*Ax[idx_zero]+c)
		du[idx_zero] = -y[idx_zero]*(2*a*Ax[idx_zero]+b) 
	else:
		f_poly = 0	
	if np.shape(idx_non_zero)[0] > 0:
		f_log = -y[idx_non_zero]*np.log(Ax[idx_non_zero])
		du[idx_non_zero] = -y[idx_non_zero]/Ax[idx_non_zero]
	else:
		f_log = 0
	indicators = np.ones([N], dtype=np.float32); Ax1 = Ax; 
	indicators[idx_zero] = 0.0; Ax1[idx_zero] =  0
	
	df = At(np.reshape(du+indicators,[H,W]))
	df = np.reshape(df,[N])+rho*(x-x0)
	f = np.sum(np.ravel(f_poly)) + np.sum(np.ravel(f_log)) + np.sum(Ax1) + 0.5*rho*np.linalg.norm(x-x0,2)**2
	return f, df

def ffd_wrapper(x, sigma, net):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	x_min, x_max = np.min(np.ravel(x)), np.max(np.ravel(x))
	
	scale_range = 1.0
	scale_shift = (1-scale_range)*0.5
	
	x1 = (x-x_min)/x_max
	x2 = x1*scale_range + scale_shift
	sigma2 = sigma*scale_range/x_max

	xt = img_to_tens(x1).to(device)
	sigmat = scalar_to_tens(sigma2).to(device)

	with torch.no_grad(): 
		noise_t = net(xt, sigmat)
		yt = xt-noise_t
	y2 = np.clip(np.squeeze(np.squeeze(yt.cpu().detach().numpy())),0,1)
	
	y1 = (y2-scale_shift)/scale_range
	y = y1*x_max+x_min
	return y

def dncnn_wrapper(x, sigma, net):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	x_min, x_max = np.min(np.ravel(x)), np.max(np.ravel(x))
	scale = 1.0
	shift = (1-scale)*0.5


	x1 = (x-x_min)/x_max
	x2 = x1*scale + shift

	xt = img_to_tens(x2).type(torch.FloatTensor).to(device)
	with torch.no_grad(): 
		yt = net(xt)
	y2 = np.clip(np.squeeze(np.squeeze(yt.cpu().detach().numpy())),0,1)

	y1 =(y2-shift)/scale
	y = y1*x_max+x_min
	return y

def realsn_wrapper(x, sigma, net):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	x_min, x_max = np.min(np.ravel(x)), np.max(np.ravel(x))
	scale = 1.0+sigma/2.0
	shift = (1-scale)*0.5


	x1 = (x-x_min)/x_max
	x2 = x1*scale + shift

	xt = img_to_tens(x2).type(torch.FloatTensor).to(device)
	with torch.no_grad(): 
		noise_t = net(xt)
		yt = xt - noise_t
	y2 = np.clip(np.squeeze(np.squeeze(yt.cpu().detach().numpy())),0,1)

	y1 =(y2-shift)/scale
	y = y1*x_max+x_min
	return y

def bm3d_wrapper(x, sigma):
	x_min, x_max = np.min(np.ravel(x)), np.max(np.ravel(x))
	x1 = (x-x_min)/x_max
	sigma1 = sigma/x_max


	y1 = bm3d(x1, sigma1)

	y = y1*x_max+x_min
	return y

def pnp_poisson(y, kernel, M, net, denoiser, rho0 = 100):
	verbose = True
	MAX_ITERS = 150
	H, W = np.shape(y)
	k_pad, k_fft = psf_to_otf(kernel, [H,W])

	A = lambda x : np.real(ifft2(fft2(x)*k_fft))
	At = lambda x : np.real(ifft2(fft2(x)*np.conj(k_fft))) 
	A_m = lambda x : M*A(x); At_m = lambda x : M*At(x) 
	
	lambda_r = 5.0/M; rho = rho0
	params = {
	'y': y,
	'A': A_m,
	'At': At_m,
	'rho': rho,
	'x0': np.zeros([H*W],  dtype=np.float32)
	}
	# Initialize x through Wiener deconvolution
	x = np.clip(l2_deconv(y/M, k_fft, 1/M),0,1)
	v = x.copy()
	u = np.zeros([H,W], dtype=np.float32)
	delta = 0.0
	for iters in range(MAX_ITERS):
		x_prev, v_prev, u_prev = x, v, u
		
		# L-BFGS solver for data subproblem
		xhat = np.reshape(v -  u, [H*W])
		params['rho'] = rho; params['x0'] = xhat
		x, f, dict_opt  = l_bfgs(func = x_subp, x0 = xhat, fprime=None, args=(params,), approx_grad=0)
		x = np.reshape(x,[H,W])
		
		# Denoising step
		vhat = x + u
		sigma = np.sqrt(lambda_r/rho)

		if denoiser == 'BM3D':
			v = bm3d_wrapper(vhat, sigma)
		elif denoiser == 'DnCNN':
			v = dncnn_wrapper(vhat, sigma, net)
		elif denoiser == 'RealSN_DnCNN':
			v = realsn_wrapper(vhat, sigma, net)
		# Scaled lagrangian update
		u = u + x - v


		rel_diff_v = np.linalg.norm(v-v_prev,'fro')/np.sqrt(H*W)
		rel_diff_x = np.linalg.norm(x-x_prev,'fro')/np.sqrt(H*W)
		rel_diff_u = np.linalg.norm(u-u_prev,'fro')/np.sqrt(H*W)
		delta_prev = delta
		delta = (1/3)*(rel_diff_x + rel_diff_v + rel_diff_u)
		
		if verbose:	print('Iteration: ', (iters+1))
		if delta > 0.99*delta_prev:
			rho *= 1.01
			if verbose:	print('Rho updated to %0.3f'%(rho))
		else:
			if verbose:	print('rho constant at %0.3f'%(rho))
		if verbose:	print('Relative Differences: %0.4f, %0.4f, %0.4f'%(rel_diff_x, rel_diff_v, rel_diff_u))
		if delta < 1e-3:
			break
	return x


def pnp_poisson_3split(y, kernel, M, net, denoiser, rho0 = 1000):
	verbose = True
	MAX_ITERS = 150
	H, W = np.shape(y)
	k_pad, k_fft = psf_to_otf(kernel, [H,W])

	A = lambda x : np.real(ifft2(fft2(x)*k_fft))
	At = lambda x : np.real(ifft2(fft2(x)*np.conj(k_fft))) 
	
	lambda_r, rho1, rho2 = 1.0, rho0, rho0
	# Initialize x through Wiener deconvolution
	x = np.clip(l2_deconv(y/M, k_fft, 1/M),0,1)
	z = x.copy()
	u = y.copy()
	v1 = np.zeros([H,W], dtype=np.float32)
	v2 = np.zeros([H,W], dtype=np.float32)
	delta =  np.inf
	for iters in range(MAX_ITERS):
		x_prev, z_prev, u_prev, v1_prev, v2_prev = x, z, u, v1, v2
		
		# Poisson Proximal step
		u0 = A(x) + v1
		u = poisson_proximal_3split(u0, y, M, rho1)
		# Denoising step
		zhat = x + v2
		sigma = np.sqrt(lambda_r/rho2)
		if denoiser == 'BM3D':
			z = bm3d_wrapper(zhat, sigma)
		elif denoiser == 'DnCNN':
			z = dncnn_wrapper(zhat, sigma, net)
		elif denoiser == 'RealSN_DnCNN':
			z = realsn_wrapper(zhat, sigma, net)
		# Deblurring step
		x0 = At(u - v1); x1 = z - v2 
		x = deblurring_3split(x0, x1, rho1, rho2, k_fft)	
		# Scaled lagrangian update
		v1 = v1 + A(x) - u
		v2 = v2 + x - z
		

		rel_diff_x = np.linalg.norm(x-x_prev,'fro')/np.sqrt(H*W)
		rel_diff_z = np.linalg.norm(z-z_prev,'fro')/np.sqrt(H*W)
		rel_diff_u = np.linalg.norm(u-u_prev,'fro')/np.sqrt(H*W)
		rel_diff_v1 = np.linalg.norm(v1-v1_prev,'fro')/np.sqrt(H*W)
		rel_diff_v2 = np.linalg.norm(v2-v2_prev,'fro')/np.sqrt(H*W)
		delta_prev = delta
		delta = (1/5)*(rel_diff_x + rel_diff_z + rel_diff_u + rel_diff_v1 + rel_diff_v2)
		
		if verbose:	print('Iteration: ', (iters+1))
		if verbose:	print('Relative Differences: %0.3f, %0.3f, %0.3f, %0.3f, %0.3f'%(rel_diff_x, rel_diff_z, rel_diff_u, rel_diff_v1, rel_diff_v2))
		if delta < 1e-3:
			break
	return x

