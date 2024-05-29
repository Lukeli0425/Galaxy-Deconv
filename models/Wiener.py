import torch
import torch.nn as nn
from torch.fft import fftn, ifftn
from utils.utils_torch import conv_fft_batch, psf_to_otf, crop_half, pad_double

class Wiener(nn.Module):
	def __init__(self):
		super(Wiener, self).__init__()
		
	def forward(self, y, psf, alpha):
		device = torch.device("cuda:0" if y.is_cuda else "cpu")
  
		_, H = psf_to_otf(psf, y.size())
		H = H.to(device)
		Ht, HtH = torch.conj(H), torch.abs(H)**2
		numerator = Ht * fftn(y, dim=[2,3])
		divisor = HtH + 350/alpha
		x = torch.real(ifftn(numerator/divisor, dim=[2,3]))
  
		return x
	
if __name__ == "__main__":
	model = Wiener()