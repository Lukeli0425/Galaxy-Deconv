import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift, fftn, ifftn


class Tikhonov(nn.Module):
	def __init__(self):
		super(Tikhonov, self).__init__()
	
	def forward(self, y, psf, lam):
		psf = psf/psf.sum() # normalize PSF
		H = fftn(psf, dim=[2,3])
		numerator = torch.conj(H) * fftn(y, dim=[2,3])
		divisor = H.abs() ** 2 + lam
		x = fftshift(ifftn(numerator/divisor, dim=[2,3])).real
		return x


class SeparableConv2d(nn.Module):
	def __init__(self, in_channels, out_channels=12, kernel_size=3, stride=1, padding=0, dilation=1,bias=False):
		super(SeparableConv2d, self).__init__()
		self.depthewise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
									kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
									groups=in_channels, bias=bias)
		self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
								   kernel_size=1, stride=1, padding=0, dilation=1,
								   groups=1, bias=bias)
	
	def forward(self, x):
		return self.pointwise(self.depthewise(x))


class DenseBlock(nn.Module):
	def __init__(self, num_layers, in_channels, growth_rate=12, kernel_size=3, skip_connection=False):
		super(DenseBlock, self).__init__()
		self.skip_connection = skip_connection
		layers = []
		channel = in_channels
		for i in range(num_layers):
			layers.append(nn.Sequential(
				nn.BatchNorm2d(channel),
				nn.ReLU(inplace=True),
				SeparableConv2d(in_channels=channel, out_channels=growth_rate, kernel_size=kernel_size)
			))
			channel += growth_rate
		self.net = nn.Sequential(*layers)
		
	def forward(self, x):
		y = x
		for layer in self.net:
			out = layer(y)
			y = torch.cat((out, y), dim=1)
		if self.skip_connection:
			y = torch.cat((x, y), dim=1)
		return y


class Down(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Down, self).__init__()
		self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

	def forward(self, x):
		return self.net(x)


class Up(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Up, self).__init__()
		self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Upsample(size=(2,2), mode='nearest')
        )

	def forward(self, x):
		return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False),
            DenseBlock(num_layers=4, in_channels=32, growth_rate=12, kernel_size=3, skip_connection=True)
        ) # [B,112,48,48]
        self.down1 = nn.Sequential(
            Down(in_channels=112, out_channels=80),
            DenseBlock(num_layers=5, in_channels=80, growth_rate=12, kernel_size=3, skip_connection=True)
        ) # [B,220,24,24]
        self.down2 =  nn.Sequential(
            Down(in_channels=220, out_channels=140),
            DenseBlock(num_layers=6, in_channels=140, growth_rate=12, kernel_size=3, skip_connection=True)
        ) # [B,352,12,12]
        self.body = nn.Sequential(
            Down(in_channels=352, out_channels=212),
            DenseBlock(num_layers=7, in_channels=212, growth_rate=12, kernel_size=3, skip_connection=False),
            Up(in_channels=294, out_channels=84)
        ) # [B,84,12,12]
        self.up1 = nn.Sequential(
            DenseBlock(num_layers=6, in_channels=436, growth_rate=12, kernel_size=3, skip_connection=False),
            Up(in_channels=508, out_channels=72)
        ) # [B,72,24,24]
        self.up2 = nn.Sequential(
            DenseBlock(num_layers=5, in_channels=292, growth_rate=12, kernel_size=3, skip_connection=False),
            Up(in_channels=352, out_channels=60)
        ) # [B,60,48,48]
        self.output = nn.Sequential(
            DenseBlock(num_layers=4, in_channels=172, growth_rate=12, kernel_size=3, skip_connection=False),
            nn.Conv2d(in_channels=220, out_channels=1, kernel_size=1, padding=0, bias=False)
        ) # [B,1,48,48]
        
    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.body(x3)
        x4 = torch.cat((x3, x4), dim=1)
        x5 = self.up1(x4)
        x5 = torch.cat((x2, x5), dim=1)
        x6 = self.up2(x5)
        x6 = torch.cat((x1, x6), dim=1)
        y = self.output(x6)
        return y


class Tikhonet(nn.Module):
	def __init__(self):
		super(Tikhonet, self).__init__()
		self.tikhonov = Tikhonov()
		self.denoiser = UNet()
		
	def forward(self, y, psf, alpha):
		device = torch.device("cuda:0" if y.is_cuda else "cpu")
		lam = torch.Tensor(0.05, requires_grad=True, device=device)	
		x = self.tikhonov(y, psf, lam)
		x = self.denoiser(x)
		return x


if __name__ == '__main__':
	model = Tikhonet()
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %s" % (total))
	