import os
import numpy as np
import matplotlib.pyplot as plt
import galsim


def get_atmos_fwhm(rng, fwhms, freq):
    freq = np.array([ 0.,   20.,  17.,  13.,  9.,   0.   ])
    freq = freq/freq.sum()
    
    if 0 <= freq and freq < 0.2:
        fwhm = 0.45
    elif 0.2 <= freq and freq < 0.37:
        fwhm = 0.55
    elif 0.37 <= freq and freq < 0.5:
        fwhm = 0.65
    elif 0.5 <= freq and freq < 0.65:
        fwhm = 0.75
    elif 0.65 <= freq and freq < 0.8:
        fwhm = 0.85
    elif 0.8 <= freq and freq < 1:
        fwhm = 0.95    
    
    return fwhm
    
    
    # x: [ 0.45, 0.55, 0.65, 0.75, 0.85, 0.95 ]
    # f: [ 0.,   20.,  17.,  13.,  9.,   0.   ]
    
if __name__ == '__main__':
    rng = galsim.BaseDeviate(seed=19)
    fwhms = np.array([ 0.45, 0.55, 0.65, 0.75, 0.85, 0.95 ])
    freq = np.array([ 0., 20., 17., 13., 9., 0.])
    table = galsim.LookupTable(x=fwhms, f=freq, interpolant='linear')
    # freq = freq/freq.sum()
    # fwhm = get_atmos_fwhm(rng, fwhms, freq)
    gs = np.linspace(0.01,0.05,50)
    plt.subplot(1,2,1)
    plt.plot(gs, gs)
    plt.ylim(0)
    
    d = galsim.DistDeviate(seed=rng, function=lambda x: x, x_min=0.01, x_max=0.05)
    fwhms = [d() for i in range(30000)]
    plt.subplot(1,2,2)
    plt.hist(fwhms, bins=50)
    
    plt.show()
    