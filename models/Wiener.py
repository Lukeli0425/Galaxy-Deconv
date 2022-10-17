import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftshift, ifftshift

class Wiener(nn.Module):
    def __init__(self):
        super(Wiener, self).__init__()
        
    def forward(self, y, psf, snr):
        
        psf = psf/psf.sum() # normalize PSF
        x = fftshift(ifft2(fft2(y) / fft2(psf) / (1+1/(fft2(psf).abs()**2 * snr)) )).real
        return x
    
if __name__ == "__main__":
    model = Wiener()