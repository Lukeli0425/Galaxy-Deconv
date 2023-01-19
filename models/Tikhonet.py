import torch
import torch.nn as nn
from torch.fft import fftn, ifftn
from utils.utils_torch import conv_fft_batch, psf_to_otf, laplacian_kernel
from models.XDenseUNet import XDenseUNet


class Tikhonov(nn.Module):
	def __init__(self, filter='Identity'):
		super(Tikhonov, self).__init__()
		self.filter = filter
        
	def forward(self, y, psf, alpha, lam):
		device = torch.device("cuda:0" if y.is_cuda else "cpu")
  
		_, H = psf_to_otf(psf, y.size())
		H = H.to(device)
		Ht, HtH = torch.conj(H), torch.abs(H)**2
		# numerator = fftn(conv_fft_batch(Ht, y/alpha), dim=[2,3])
		numerator = Ht * fftn(y/alpha, dim=[2,3])
		if self.filter == 'Identity':
			divisor = HtH + lam/alpha
		elif self.filter == 'Laplacian':
			lap = laplacian_kernel()
			_, L = psf_to_otf(lap, y.size())
			LtL = torch.abs(L.to(device)) ** 2
			divisor = HtH + lam * LtL / alpha
		x = torch.real(ifftn(numerator/divisor, dim=[2,3]))

		return x


class Tikhonet(nn.Module):
	def __init__(self, filter='Identity'):
		super(Tikhonet, self).__init__()
		self.tikhonov = Tikhonov(filter=filter)
		self.denoiser = XDenseUNet()
		self.lam = torch.tensor(1., requires_grad=True) # Learnable parameter.
		
	def forward(self, y, psf, alpha):
		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		x = self.tikhonov(y, psf, alpha, self.lam)
		x = self.denoiser(x)
  
		return x * alpha


if __name__ == '__main__':
	model = Tikhonet()
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %s" % (total))
	