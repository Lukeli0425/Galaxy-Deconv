import torch
import torch.nn as nn
from utils.utils_torch import conv_fft_batch, psf_to_otf

class Richard_Lucy(nn.Module):
    def __init__(self, n_iters):
        super(Richard_Lucy, self).__init__()
        self.n_iters = n_iters
        
    def forward(self, y, psf):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ones = torch.ones_like(y)
        _, H = psf_to_otf(psf, y.size())
        H = H.to(device)
        Ht = torch.conj(H).to(device)
        x = y.clone() # initial guess
        for i in range(self.n_iters):
            Hx = conv_fft_batch(H, x).to(device)
            numerator = conv_fft_batch(Ht, y/Hx)
            divisor = conv_fft_batch(H, ones)
            x = x*numerator/divisor
        return x
    
if __name__ == "__main__":
    model = Richard_Lucy()
