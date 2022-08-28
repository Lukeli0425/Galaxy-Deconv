import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.ResUNet import ResUNet

def conv2d_from_kernel(kernel, channels, device=None, stride=1):
    """
    Returns nn.Conv2d and nn.ConvTranspose2d modules from 2D kernel, such that 
    nn.ConvTranspose2d is the adjoint operator of nn.Conv2d
    Arg:
        kernel: 2D kernel
        channels: number of image channels
    """
    kernel_size = kernel.shape
    kernel = kernel/kernel.sum()
    kernel = kernel.repeat(channels, 1, 1, 1)
    # print(kernel.shape)

    filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        # padding=kernel_size//2
    )
    filter.weight.data = kernel
    filter.weight.requires_grad = False

    filter_adjoint = nn.ConvTranspose2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        # padding=kernel_size//2,
    )
    filter_adjoint.weight.data = kernel
    filter_adjoint.weight.requires_grad = False

    # return filter.to(device), filter_adjoint.to(device)
    return filter, filter_adjoint

class PnP_ADMM(nn.Module):
    def __init__(self, n_iters=8, max_cgiter=3, cg_tol=1e-7, step_size=2e-4):
        super(PnP_ADMM, self).__init__()
        self.n_iters = n_iters
        self.max_cgiter = max_cgiter
        self.cg_tol = cg_tol
        self.step_size = step_size
        
        self.denoiser = ResUNet()
    
    def forward(self, x, kernel):
        kernel = nn.ReplicationPad2d((0, 1, 0, 1))(kernel).squeeze(dim=0).squeeze(dim=0)
        # print(kernel.shape)
        filter, filter_adjoint = conv2d_from_kernel(kernel, 1)
        x_h = filter_adjoint(x)

        def conjugate_gradient(A, b, x0, max_iter, tol):
            """
            Conjugate gradient method for solving Ax=b
            """
            x = x0
            r = b-A(x)
            d = r
            for _ in range(max_iter):
                z = A(d)
                rr = torch.sum(r**2)
                alpha = rr/torch.sum(d*z)
                x = x + alpha*d
                r = r - alpha*z
                if torch.norm(r)/torch.norm(b) < tol:
                    break
                beta = torch.sum(r**2)/rr
                d = r + beta*d        
            return x

        def cg_leftside(x):
            """
            Return left side of Ax=b, i.e., Ax
            """
            return filter_adjoint(filter(x)) + self.step_size*x

        def cg_rightside(x):
            """
            Returns right side of Ax=b, i.e. b
            """
            return x_h + self.step_size*x
        
        x = torch.zeros_like(x_h)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        
        for idx in range(self.n_iters):
            b = cg_rightside(v-u)
            x = conjugate_gradient(cg_leftside, b, x, self.max_cgiter, self.cg_tol)
            v = self.denoiser(x+u)
            u = u + (x - v)
        return v[:,:,24:72,24:72]
