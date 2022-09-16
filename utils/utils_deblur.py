import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.signal import convolve2d


def pad(h, shape_x):
	shape_h = np.shape(h)
	offset = 1
	hpad = np.zeros(shape_x, dtype=np.float32)
	i1, j1 = np.int32((shape_x[0] - shape_h[0])/2)+offset, np.int32((shape_x[1] - shape_h[1])/2)+offset 
	i2, j2 = i1 + shape_h[0], j1 + shape_h[1]
	hpad[i1:i2, j1:j2] = h
	return hpad

def shrinkage(z, beta):
	c1, c2 = z -beta,  z + beta
	z_out = np.clip(c1,0,np.inf) + np.clip(z+beta,-np.inf,0)
	return z_out

# def crop(img, M, N):
# 	# M < N
# 	i1, i2 = np.int32(0.5*(N-M)), np.int32(0.5*(M+N))
# 	return img[i1:i2, i1:i2]

def crop(h, shape_crop):
	shape_h = np.shape(h)
	i1, j1 = np.int32((shape_h[0] - shape_crop[0])/2), np.int32((shape_h[1] - shape_crop[1])/2) 
	i2, j2 = np.int32((shape_h[0] + shape_crop[0])/2), np.int32((shape_h[1] + shape_crop[1])/2) 
	return h[i1:i2, j1:j2]


def gauss_kernel(size, sigma):
	ax = np.linspace(-(size-1)*0.5, size*0.5, size)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.exp( -(xx**2 + yy**2)/(2*(sigma**2)) )
	kernel = kernel/np.sum(kernel.ravel())
	return kernel

def disk(size, r):
	ax = np.linspace(-(size-1)*0.5, size*0.5)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.asarray((xx**2 + yy**2) < r**2, dtype=np.float32)
	kernel = kernel/np.sum(kernel.ravel())
	return kernel


def D(U):
	Dux1 = np.diff(U,n = 1,axis =1)
	Dux = np.zeros(np.shape(U))
	Dux[:,0:-1] = Dux1.copy()
	Dux[:,-1] = U[:,0] - U[:,-1]

	Duy1 = np.diff(U, n=1, axis=0)
	Duy = np.zeros(np.shape(U))
	Duy[0:-1,:] = Duy1.copy()
	Duy[-1,:] = U[0,:] - U[-1,:]

	return Dux, Duy

def Mask(Dx, Dy, tau_s =0.1, tau_r =0.1):
	g = (1/25)*np.ones([5,5], dtype=np.float32)

	Dxy = np.sqrt(Dx**2 + Dy**2)
	a, b = convolve2d(Dx,g,mode='same'), convolve2d(Dy,g,mode='same')
	c = convolve2d(Dxy,g,mode='same')
	R = np.sqrt(a**2 + b**2)/(c +0.5) 
	M = np.max(R-tau_r, 0)
	Dx1, Dy1 = Dx*np.max(M*Dxy-tau_s,0), Dy*np.max(M*Dxy-tau_s,0)
	return  Dx1, Dy1, M

def k_ifft(x_rec, y, lambda_l2, lambda_l1, M=25):
	# M = kernel size
	# Solves ||Delta(x)*k - Delta(y)||^2 + \lambda_s*||k||^2
	# Then passes through a TV denoiser
	N = np.shape(x_rec)[0]
	Dx_1, Dx_2 = D(x_rec)
	Dx_1, Dx_2, _ = Mask(Dx_1, Dx_2)
	Dy_1, Dy_2 = D(y)
	Dy_1, Dy_2, _ = Mask(Dy_1, Dy_2)

	num = np.conj(fft2(Dx_1))*fft2(Dy_1) + np.conj(fft2(Dx_2))*fft2(Dy_2)
	den = np.abs(fft2(Dx_1))**2 + np.abs(fft2(Dx_2))**2 + lambda_l2
	k0 = np.real(ifftshift(ifft2(num/den)))	
	k0 = np.clip(crop(k0,M,N),0,np.inf)
	k0 = k0/np.sum(np.ravel(k0))


	return k0

def rgb_to_bayer(x):
	H, W, _ = np.shape(x)
	x_bayer = np.zeros([2*H, 2*W])
	x_r, x_g, x_b =  x[:,:,0], x[:,:,1], x[:,:,2]
	
	x_bayer[0::2,0::2] = x_r
	x_bayer[0::2,1::2] = x_g
	x_bayer[1::2,0::2] = x_g
	x_bayer[1::2,1::2] = x_b

	return x_bayer


def rggb_to_rgb(x_list, switch_rgb):
	H, W = np.shape(x_list[0])
	x_rgb = np.zeros([H, W, 3])
	
	x_rgb[:,:,0] = x_list[0]
	x_rgb[:,:,1] = (x_list[1]+x_list[2])*0.5
	x_rgb[:,:,2] = x_list[3]
	
	if switch_rgb:
		x_rgb = np.flip(x_rgb,2)
	return x_rgb


def psf2otf(kernel, size):
	psf = np.zeros(size,dtype=np.float32)
	centre =  np.shape(kernel)[0]//2 + 1
	
	psf[:centre, :centre] = kernel[(centre-1):,(centre-1):]
	psf[:centre, -(centre-1):] = kernel[(centre-1):, :(centre-1)]
	psf[-(centre-1):, :centre] = kernel[:(centre-1), (centre-1):]
	psf[-(centre-1):, -(centre-1):] = kernel[:(centre-1),:(centre-1)]
	
	otf = fft2(psf, size)
	return psf, otf