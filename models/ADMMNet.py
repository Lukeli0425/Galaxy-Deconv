import torch
import torch.fft as tfft
import torch.nn as nn
from torch.autograd import Variable

from models.ResUNet import ResUNet
from models.XDenseUNet import XDenseUNet
from utils.utils_torch import conv_fft, conv_fft_batch, psf_to_otf


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
	def __init__(self, model_file):
		super(Z_Update_ResUNet, self).__init__()
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")	
		self.net = ResUNet()
		try:
			self.net.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
		except:
			raise ValueError('Please provide a valid model file for ResUNet denoiser.')

	def forward(self, z):
		z_out = self.net(z.float())
		return z_out


class Z_Update_XDenseUNet(nn.Module):
	"""Updating Z with XDenseUNet as denoiser."""
	def __init__(self, model_file):
		super(Z_Update_XDenseUNet, self).__init__()		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")	
		self.net = XDenseUNet()
		self.net.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
  
	def forward(self, z):
		z_out = self.net(z.float())
		return z_out


class ADMMNet(nn.Module):
	def __init__(self, n_iters=8, llh='Poisson', denoiser='ResUNet', PnP=True, model_file=None):
		super(ADMMNet, self).__init__()
		self.n = n_iters # Number of iterations.
		self.llh = llh
		self.PnP = PnP
		self.denoiser = denoiser
		self.X = X_Update() # FFT based quadratic solution.
		self.V = V_Update_Poisson() if llh=='Poisson' else V_Update_Gaussian() # Poisson/Gaussian MLE.
		self.Z = (Z_Update_ResUNet(model_file=model_file) if self.denoiser=='ResUNet' else Z_Update_XDenseUNet(model_file=model_file)) if PnP else Z_Update() # Denoiser.	

  
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
		x = self.init_l2(y, H, alpha) # Initialization using Wiener Deconvolution
		x_list.append(x)
		# Other ADMM variables
		z = Variable(x.data.clone()).to(device)
		v = Variable(y.data.clone()).to(device)
		u1 = torch.zeros(y.size()).to(device)
		u2 = torch.zeros(y.size()).to(device)
		
        # ADMM iterations
		for n in range(self.n):
			rho1 = 0.5
			rho2 = 0.5
			# V, Z and X updates
			v = self.V(conv_fft_batch(H,x) + u2, y, rho2, alpha) if self.llh=='Poisson' else self.V(conv_fft_batch(H,x) + u2, y/alpha, rho2)
			z = self.Z(x + u1) if self.PnP else self.Z(x + u1, lam, rho1)
			x = self.X(z - u1, conv_fft_batch(Ht,v - u2), HtH, rho1, rho2)
			# Lagrangian updates
			u1 = u1 + x - z			
			u2 = u2 + conv_fft_batch(H,x) - v
			x_list.append(x)

		return x_list[-1] * alpha # if self.llh=='Poisson' else x_list[-1]


if __name__ == '__main__':
	model = ADMMNet()
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %s" % (total))