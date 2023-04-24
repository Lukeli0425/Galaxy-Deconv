import math

import numpy as np
import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.ResUNet import ResUNet
from models.XDenseUNet import XDenseUNet
from utils.utils_torch import conv_fft, conv_fft_batch, psf_to_otf


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		# nn.init.uniform(m.weight.data, 1.0, 0.02)
		m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
		nn.init.constant(m.bias.data, 0.0)


class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(DoubleConv, self).__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""
	def __init__(self, in_channels, out_channels):
		super(Down, self).__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class InitNet(nn.Module):
	def __init__(self, n):
		super(InitNet, self).__init__()
		self.n = n
		self.conv_layers = nn.Sequential(
			Down(1,4),
			Down(4,8),
			Down(8,16),
			Down(16,16))
		self.mlp = nn.Sequential(
			nn.Linear(16*8*8+1, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 2*self.n),
			nn.Softplus())
		self.resize = nn.Upsample(size=[256,256], mode='bilinear', align_corners=True)
		
	def forward(self, kernel, alpha):
		N, _, h, w  = kernel.size()
		h1, h2 = int(np.floor(0.5*(128-h))), int(np.ceil(0.5*(128-h)))
		w1, w2 = int(np.floor(0.5*(128-w))), int(np.ceil(0.5*(128-w)))
		k_pad = F.pad(kernel, (w1,w2,h1,h2), "constant", 0)
		H = tfft.fftn(k_pad,dim=[2,3])
		HtH = torch.abs(H)**2
		x = self.conv_layers(HtH.float())
		x = torch.cat((x.view(N,1,16*8*8),  alpha.float().view(N,1,1)), axis=2).float()
		output = self.mlp(x)+1e-6

		rho1_iters = output[:,:,0:self.n].view(N, 1, 1, self.n)
		rho2_iters = output[:,:,self.n:2*self.n].view(N, 1, 1, self.n)
		return rho1_iters, rho2_iters


class X_Update(nn.Module):
	def __init__(self):
		super(X_Update, self).__init__()

	def forward(self, x0, x1, HtH, rho1, rho2):
		lhs = rho1*HtH + rho2 
		rhs = tfft.fftn(rho1*x0 + rho2*x1, dim=[2,3] )
		x = tfft.ifftn(rhs/lhs, dim=[2,3])
		return x.real


class V_Update_Poisson(nn.Module):
	def __init__(self):
		super(V_Update_Poisson, self).__init__()

	def forward(self, v_tilde, y, rho2, alpha):
		t1 = rho2*v_tilde - alpha 
		return 0.5*(1/rho2)*(-t1 + torch.sqrt(t1**2 + 4*y*rho2))


class V_Update_Gaussian(nn.Module):
	def __init__(self):
		super(V_Update_Gaussian, self).__init__()

	def forward(self, v_tilde, y, rho2):
		return (rho2*v_tilde + y)/(1+rho2)


class Z_Update(nn.Module):
	"""Updating Z with l1 norm."""
	def __init__(self):
		super(Z_Update, self).__init__()		

	def forward(self, z_tilde, lam, rho1):
		z_out = torch.sign(z_tilde) * torch.max(torch.zeros_like(z_tilde), torch.abs(z_tilde) - lam/rho1)
		return z_out


class Z_Update_ResUNet(nn.Module):
	"""Updating Z with ResUNet as denoiser."""
	def __init__(self):
		super(Z_Update_ResUNet, self).__init__()		
		self.net = ResUNet()

	def forward(self, z):
		z_out = self.net(z.float())
		return z_out


class Z_Update_XDenseUNet(nn.Module):
	"""Updating Z with XDenseUNet as denoiser."""
	def __init__(self):
		super(Z_Update_XDenseUNet, self).__init__()		
		self.net = XDenseUNet()

	def forward(self, z):
		z_out = self.net(z.float())
		return z_out


class Unrolled_ADMM(nn.Module):
	def __init__(self, n_iters=8, llh='Poisson', denoiser='ResUNet', PnP=True, SubNet=True):
		super(Unrolled_ADMM, self).__init__()
		self.n = n_iters # Number of iterations.
		self.llh = llh
		self.PnP = PnP
		self.SubNet = SubNet
		self.denoiser = denoiser
		self.X = X_Update() # FFT based quadratic solution.
		self.V = V_Update_Poisson() if llh=='Poisson' else V_Update_Gaussian() # Poisson/Gaussian MLE.
		self.Z = (Z_Update_ResUNet() if self.denoiser=='ResUNet' else Z_Update_XDenseUNet()) if PnP else Z_Update() # Denoiser.	
		if self.SubNet:
			self.init = InitNet(self.n)
		else:
			self.rho1_iters = torch.ones(size=[self.n,], requires_grad=True)
			self.rho2_iters = torch.ones(size=[self.n,], requires_grad=True)
  
	def init_l2(self, y, H, alpha):
		Ht, HtH = torch.conj(H), torch.abs(H)**2
		rhs = tfft.fftn( conv_fft_batch(Ht, y/alpha), dim=[2,3] )
		lhs = HtH + (1/alpha)
		x0 = torch.real(tfft.ifftn(rhs/lhs, dim=[2,3]))
		x0 = torch.clamp(x0,0,1)
		return x0

	def forward(self, y, kernel, alpha):
		device = torch.device("cuda:0" if y.is_cuda else "cpu")
		x_list = []
		N, _, _, _ = y.size()
		y = torch.max(y, torch.zeros_like(y))
  
		# Generate auxiliary variables for convolution
		k_pad, H = psf_to_otf(kernel, y.size())
		H = H.to(device)
		Ht, HtH = torch.conj(H), torch.abs(H)**2
		if self.SubNet:
			rho1_iters, rho2_iters = self.init(kernel, alpha) 	# Hyperparameters
		x = self.init_l2(y, H, alpha) # Initialization using Wiener Deconvolution
		x_list.append(x)
		# Other ADMM variables
		z = Variable(x.data.clone()).to(device)
		v = Variable(y.data.clone()).to(device)
		u1 = torch.zeros(y.size()).to(device)
		u2 = torch.zeros(y.size()).to(device)
		
        # ADMM iterations
		for n in range(self.n):
			if self.SubNet:
				rho1 = rho1_iters[:,:,:,n].view(N,1,1,1)
				rho2 = rho2_iters[:,:,:,n].view(N,1,1,1)
			else:
				rho1, rho2 = self.rho1_iters[n], self.rho2_iters[n]
			# V, Z and X updates
			v = self.V(conv_fft_batch(H,x) + u2, y, rho2, alpha) if self.llh=='Poisson' else self.V(conv_fft_batch(H,x) + u2, y/alpha, rho2)
			z = self.Z(x + u1) if self.PnP else self.Z(x + u1, lam, rho1)
			x = self.X(z - u1, conv_fft_batch(Ht,v - u2), HtH, rho1, rho2)
			# Lagrangian updates
			u1 = u1 + x - z			
			u2 = u2 + conv_fft_batch(H,x) - v
			x_list.append(x)

		return x_list[-1] * alpha if self.llh=='Poisson' else x_list[-1]



if __name__ == '__main__':
	model = Unrolled_ADMM()
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %s" % (total))