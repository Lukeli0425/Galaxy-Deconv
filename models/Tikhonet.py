import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from models.ResUNet import ResUNet


class Tikhonov(nn.Module):
    def __init__(self):
        super(Tikhonov, self).__init__()
    
    def forward(self, y, psf, lam):
        psf = psf/psf.sum() # normalize PSF
        H = fftn(psf, dim=[2,3])
        numerator = torch.conj(H) * fftn(y, dim=[2,3])
        divisor = H.abs() ** 2 + lam.unsqueeze(dim=1)
        x = fftshift(ifftn(numerator/divisor, dim=[2,3])).real
        return x


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


class LambdaNet(nn.Module):
	"""A subnetwork that learns lambda for Tikhonov regularization from PSF and average photon level."""
	def __init__(self):
		super(LambdaNet, self).__init__()
		self.conv_layers = nn.Sequential(
			Down(1,4),
			Down(4,8),
			Down(8,16),
			Down(16,16))
		self.mlp = nn.Sequential(
			nn.Linear(16*8*8+1, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 32),
			nn.ReLU(inplace=True),
			nn.Linear(32, 1),
			nn.Softplus())
		
	def forward(self, kernel, alpha):
		N, _, h, w  = kernel.size()
		h1, h2 = int(np.floor(0.5*(128-h))), int(np.ceil(0.5*(128-h)))
		w1, w2 = int(np.floor(0.5*(128-w))), int(np.ceil(0.5*(128-w)))
		k_pad = F.pad(kernel, (w1,w2,h1,h2), "constant", 0)
		H = fftn(k_pad, dim=[2,3])
		HtH = torch.abs(H)**2
  
		x = self.conv_layers(HtH.float())
		x = torch.cat((x.view(N,1,16*8*8), alpha.float().view(N,1,1)), axis=2).float()
		lam = self.mlp(x) + 1e-6
		return lam


class Tikhonet(nn.Module):
    def __init__(self):
        super(Tikhonet, self).__init__()
        self.init = LambdaNet()
        self.tikhonov = Tikhonov()
        self.denoiser = ResUNet(nc=[16, 32, 64, 128], nb=1)
        
    def forward(self, y, psf, alpha):
        lam = self.init(psf, alpha)
        
        x = self.tikhonov(y, psf, lam)
        x = self.denoiser(x)
        return x


if __name__ == '__main__':
    model = Tikhonet()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %s" % (total))