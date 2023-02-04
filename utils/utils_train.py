import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import utils.cadmos_lib as cl


def get_model_name(method, loss, filter='Laplacian', n_iters=8, llh='Gaussian', PnP=True, remove_SubNet=False):
    if method == 'Unrolled_ADMM':
        model_name = f'{llh}{"_PnP" if PnP else ""}_ADMM_{n_iters}iters{"_No_SubNet" if remove_SubNet else ""}' 
    elif method == 'Tikhonet' or method == 'ShapeNet':
        model_name = f'{method}_{filter}'
    else:
        model_name = method 
        
    if not method == 'ShapeNet':
        model_name = f'{model_name}_{loss}'
    
    return model_name


class MultiScaleLoss(nn.Module):
	def __init__(self, scales=3, norm='L1'):
		super(MultiScaleLoss, self).__init__()
		self.scales = scales
		if norm == 'L1':
			self.loss = nn.L1Loss()
		if norm == 'L2':
			self.loss = nn.MSELoss()

		self.weights = torch.FloatTensor([1/(2**scale) for scale in range(self.scales)])
		self.multiscales = [nn.AvgPool2d(2**scale, 2**scale) for scale in range(self.scales)]

	def forward(self, output, target):
		loss = 0
		for i in range(self.scales):
			output_i, target_i = self.multiscales[i](output), self.multiscales[i](target)
			loss += self.weights[i]*self.loss(output_i, target_i)
			
		return loss


class ShapeConstraint(nn.Module):
    def __init__(self, device, fov_pixels=48, gamma=1, n_shearlet=2):
        super(ShapeConstraint, self).__init__()
        self.mse = nn.MSELoss()
        self.gamma = gamma
        U = cl.makeUi(fov_pixels, fov_pixels)
        shearlets, shearlets_adj = cl.get_shearlets(fov_pixels, fov_pixels, n_shearlet)
        # shealret adjoint of U, i.e Psi^{Star}(U)
        self.psu = np.array([cl.convolve_stack(ui, shearlets_adj) for ui in U])
        self.mu = torch.Tensor(cl.comp_mu(self.psu))
        self.mu = torch.Tensor(self.mu).to(device)
        self.psu = torch.Tensor(self.psu).to(device)
        
    def forward(self, output, target):
        loss = self.mse(output, target)
        for i in range(6):
            for j in range(self.psu.shape[1]):
                loss += self.gamma * self.mu[i,j] * (F.l1_loss(output*self.psu[i,j], target*self.psu[i,j]) ** 2) / 2.
        return loss
    
    
if __name__ == "__main__":
    print(get_model_name('ResUNet', 'MSE'))
    
    