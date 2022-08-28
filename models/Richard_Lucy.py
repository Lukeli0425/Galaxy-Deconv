import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from utils_poisson_deblurring.utils_torch import conv_fft_batch, psf_to_otf

class Richard_Lucy(nn.Module):
    def __init__(self, n_iters):
        super(Richard_Lucy, self).__init__()
        self.n_iters = n_iters
        
    def forward(self, y, psf):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        psf = psf/psf.sum() # normalize PSF
        ones = torch.ones_like(y)
        _, H = psf_to_otf(psf, y.size())
        H = H.to(device)
        Ht = torch.conj(H).to(device)
        x = torch.ones_like(y) # initial guess
        for i in range(self.n_iters):
            Hx = conv_fft_batch(H, x).to(device)
            numerator = conv_fft_batch(Ht, y/Hx)
            divisor = conv_fft_batch(H, ones)
            x = x*numerator/divisor
        return x
    
if __name__ == "__main__":
    model = Richard_Lucy()
