import os
import json
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import galsim
from utils.utils_data import down_sample


def get_LSST_PSF(lam_over_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                 atmos_fwhm, atmos_e, atmos_beta, spher, trefoil1, trefoil2,
                 g1_err=0, g2_err=0,
                 fov_pixels=48, pixel_scale=0.2, upsample=4):
    """Simulate a PSF from a ground-based observation.

    Args:
        lam_over_diam (float): Wavelength over diameter of the telescope.
        opt_defocus (float): Defocus in units of incident light wavelength.
        opt_c1 (float): Coma along y in units of incident light wavelength.
        opt_c2 (float): Coma along x in units of incident light wavelength.
        opt_a1 (float): Astigmatism (like e2) in units of incident light wavelength. 
        opt_a2 (float): Astigmatism (like e1) in units of incident light wavelength. 
        opt_obscuration (float): Linear dimension of central obscuration as fraction of pupil linear dimension, [0., 1.).
        atmos_fwhm (float): The full width at half maximum of the Kolmogorov function for atmospheric PSF.
        atmos_e (float): Ellipticity of the shear to apply to the atmospheric component.
        atmos_beta (float): Position angle (in radians) of the shear to apply to the atmospheric component, twice the phase of a complex valued shear.
        spher (float): Spherical aberration in units of incident light wavelength.
        trefoil1 (float): Trefoil along y axis in units of incident light wavelength.
        trefoil2 (float): Trefoil along x axis in units of incident light wavelength.
        g1_err (float, optional): The first component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        g2_err (float, optional): The second component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for the PSF image. Defaults to `4`.

    Returns:
        `torch.Tensor`: Simulated PSF image with shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """

    # Atmospheric PSF
    atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1)
    atmos = atmos.shear(e=atmos_e, beta=atmos_beta*galsim.radians)

    # Optical PSF
    optics = galsim.OpticalPSF(lam_over_diam, defocus = opt_defocus,
                               coma1 = opt_c1, coma2 = opt_c2,
                               astig1 = opt_a1, astig2 = opt_a2,
                               spher=spher, trefoil1=trefoil1, trefoil2=trefoil2,
                               obscuration = opt_obscuration,
                               flux=1)

    # Convolve the two components.
    psf = galsim.Convolve([atmos, optics])
    
    # Shear the overall PSF to simulate a erroneously estimated PSF when necessary.
    psf = psf.shear(g1=g1_err, g2=g2_err) 

    # Draw PSF images.
    psf_image = galsim.ImageF(fov_pixels*upsample, fov_pixels*upsample)
    psf.drawImage(psf_image, scale=pixel_scale/upsample, method='auto')
    psf_image = torch.from_numpy(psf_image.array)
         
    return psf_image

def get_Webb_PSF(fov_pixels, insts=['NIRCam', 'NIRSpec','NIRISS', 'MIRI', 'FGS']):
    """Calculate all PSFs for given JWST instruments.

    Args:
        fov_pixels (int): Width of the simulated images in pixels.
        insts (list, optional): Instruments of JWST for PSF calculation. Defaults to ['NIRCam', 'NIRSpec','NIRISS', 'MIRI', 'FGS'].

    Returns:
        tuple: List of PSF names and dictionary containing PSF images.
    """
    psfs = dict() # all PSF images
    for instname in insts:
        inst = webbpsf.instrument(instname)
        filters = inst.filter_list
        for filter in filters:
            inst.filter = filter
            try:
                logger.info(f'Calculating Webb PSF: {instname} {filter}')
                psf_list = inst.calc_psf(fov_pixels=fov_pixels, oversample=1)
                psf = torch.from_numpy(psf_list[0].data)
                psf = torch.max(torch.zeros_like(psf), psf) # set negative pixels to zero
                psf /= psf.sum()
                psfs[instname+filter] = (psf, inst.pixelscale)
            except:
                pass
    psf_names = list(psfs.keys())

    return psf_names, psfs

def get_COSMOS_Galaxy(cosmos_catalog, real_galaxy_catalog, idx, 
                      gal_g, gal_beta, theta, gal_mu, dx, dy, 
                      fov_pixels=48, pixel_scale=0.2, upsample=4):
    """Simulate a background galaxy with data from COSMOS Catalog.

    Args:
        cosmos_catalog (`galsim.COSMOSCatalog`): A `galsim.COSMOSCatalog` object, from which the parametric galaxies are read out.
        idx (int): Index of the chosen galaxy in the catalog.
        gal_flux (float): Total flux of the galaxy in the simulated image.
        sky_level (float): Skylevel in the simulated image.
        gal_g (float): The shear to apply.
        gal_beta (float): Position angle (in radians) of the shear to apply, twice the phase of a complex valued shear.
        theta (float): Rotation angle of the galaxy (in radians, positive means anticlockwise).
        gal_mu (float): The lensing magnification to apply.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for galaxy image. Defaults to `4`.

    Returns:
        `torch.Tensor`: Simulated galaxy image of shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """

    # Read out real galaxy from the catalog.
    # gal_ori = cosmos_catalog.makeGalaxy(idx, gal_type='parametric', sersic_prec=0.05) # Use parametric model
    gal_ori = galsim.RealGalaxy(real_galaxy_catalog, index = idx)
    
    # Add random rotation, shear, and magnification.
    gal = gal_ori.rotate(theta * galsim.radians) # Rotate by a random angle
    gal = gal.shear(g=gal_g, beta=gal_beta * galsim.radians) # Apply the desired shear
    gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.
    
    # Draw galaxy image.
    gal_image = galsim.ImageF(fov_pixels*upsample, fov_pixels*upsample)
    # try:
    #     gal.drawImage(gal_image, scale=pixel_scale, offset=(dx,dy), method='auto')
    # except:
    psf_hst = real_galaxy_catalog.getPSF(idx)
    gal = galsim.Convolve([psf_hst, gal]) # Concolve wth original PSF of HST.
    gal.drawImage(gal_image, scale=pixel_scale/upsample, offset=(dx,dy), method='auto')
        
    gal_image = torch.from_numpy(gal_image.array) # Convert to PyTorch.Tensor.
    
    return gal_image


def generate_data(data_path, train_split=0.7,
                  survey='LSST', I='23.5', fov_pixels=48, pixel_scale=0.2, upsample=4,
                  snrs = [10, 15, 20, 40, 60, 80, 100, 150, 200, 300],
                  shear_errs=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
                  fwhm_errs=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]):
    """Generate simulated galaxy images and corresponding PSFs for train and test dataset.

    Args:
        data_path (str): Path to save the dataset. 
        train_split (float, optional): Proportion of data used in train dataset, the rest will be used in test dataset. Defaults to 0.7.
        survey (str, optional): _description_. Defaults to 'LSST'.
        I (str, optional): The sample in COSMOS data to use, "23.5" or "25.2". Defaults to '23.5'.
        fov_pixels (int, optional):  Size of the simulated images in pixels.. Defaults to 48.
        pixel_scale (float, optional): Pixel scale in arcsec of the images. Defaults to 0.2.
        upsample (int, optional): Upsampling factor for simulations. Defaults to 4.
        snrs (list, optional): The list of SNR to be simulated for testing. Defaults to [10, 15, 20, 40, 60, 80, 100, 150, 200, 300].
        shear_errs (list, optional): The list of systematic PSF shear error to be simulated for testing. Defaults to [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2].
        fwhm_errs (list, optional): The list of systematic PSF FWHM error to be simulated for testing. Defaults to [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3].
    """
    
    logger = logging.getLogger('DataGenerator')
    logger.info('Simulating %s images using I=%s COSMOS data.', survey, I)
    
    # Create directory for the dataset.
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(os.path.join(data_path, 'obs')):
        os.mkdir(os.path.join(data_path, 'obs'))
    if not os.path.exists(os.path.join(data_path, 'gt')):
        os.mkdir(os.path.join(data_path, 'gt'))
    if not os.path.exists(os.path.join(data_path, 'psf')):
        os.mkdir(os.path.join(data_path, 'psf'))
    if not os.path.exists(os.path.join(data_path, 'visualization')): 
        os.mkdir(os.path.join(data_path, 'visualization'))

    # Read the catalog.
    try:
        real_galaxy_catalog = galsim.RealGalaxyCatalog(dir='/mnt/WD6TB/tianaoli/COSMOS_23.5_training_sample/', sample=I)
        cosmos_catalog = galsim.COSMOSCatalog(dir='/mnt/WD6TB/tianaoli/COSMOS_23.5_training_sample/', sample=I)
        n_total = real_galaxy_catalog.nobjects #- 56030
        logger.info('Successfully read in %s I=%s galaxies.', n_total, I)
    except:
        logger.warning('Failed reading in I=%s galaxies.', I)
          
    sequence = np.arange(0, n_total) # Generate random sequence for dataset.
    np.random.shuffle(sequence)
    n_train = int(train_split * n_total)
    
    info = {'survey':survey, 'I':I, 'fov_pixels':fov_pixels, 'pixel_scale':pixel_scale,
            'n_total':n_total, 'n_train':n_train, 'n_test':n_total - n_train, 'sequence':sequence.tolist()}
    info_file = os.path.join(data_path, f'info.json')
    with open(info_file, 'w') as f:
        json.dump(info, f)
    logger.info('Dataset information saved to %s.', info_file)

    # Random number generators for the parameters.
    random_seed = 37347
    rng_base = galsim.BaseDeviate(seed=random_seed)
    rng = galsim.UniformDeviate(seed=random_seed) # U(0,1).
    rng_defocus = galsim.GaussianDeviate(rng_base, mean=0., sigma=0.36) # N(0,0.36).
    rng_gaussian = galsim.GaussianDeviate(rng_base, mean=0., sigma=0.07) # N(0,0.07).
    fwhms = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    freqs = np.array([0., 20., 17., 13., 9., 0.])
    fwhm_table = galsim.LookupTable(x=fwhms, f=freqs, interpolant='spline')
    fwhms = np.linspace(fwhms[0], fwhms[-1], 100) # Upsample the distribution.
    freqs = np.array([fwhm_table(fwhm) for fwhm in fwhms]) / fwhm_table.integrate() # Normalization.
    rng_fwhm = galsim.DistDeviate(seed=rng_base, function=galsim.LookupTable(x=fwhms, f=freqs, interpolant='spline'))
    rng_gal_shear = galsim.DistDeviate(seed=rng, function=lambda x: x, x_min=0.01, x_max=0.05)
    rng_snr = galsim.DistDeviate(seed=rng, function=lambda x: 1/(x**0.41), x_min=17, x_max=320, npoints=1000) # Log-uniform Distribution.
    
    # Calculate all Webb PSFs and split for train/test
    # if survey == 'JWST':
    #     psf_names, psfs = get_Webb_PSF(fov_pixels=fov_pixels)
    #     np.random.shuffle(psf_names)
    #     train_psfs = psf_names[:int(len(psf_names) * train_split)]
    #     test_psfs = psf_names[int(len(psf_names) * train_split):]

    for k, _ in zip(range(0, n_total), tqdm(range(0, n_total))):
        idx = sequence[k] # Index of galaxy in the catalog.
        
        if survey == 'JWST': # Choose a Webb PSF 
            psf_name = np.random.choice(train_psfs) if k < n_train else np.random.choice(test_psfs)
            psf_image, pixel_scale = psfs[psf_name]
        elif survey == 'LSST': # Simulate a LSST PSF 
            # Atmospheric PSF
            atmos_fwhm = rng_fwhm()             # Atmospheric seeing (arcsec), the FWHM of the Kolmogorov function.
            atmos_e = 0.01 + 0.02 * rng()       # Ellipticity of atmospheric PSF (magnitude of the shear in the “distortion” definition), U(0.01, 0.03).
            atmos_beta = 2. * np.pi * rng()     # Shear position angle (radians), N(0,2*pi).

            # Optical PSF
            opt_defocus = rng_defocus()         # Defocus (wavelength), N(0.0.36).
            opt_a1 = rng_gaussian()             # Astigmatism (like e2) (wavelength), N(0.0.07).
            opt_a2 = rng_gaussian()             # Astigmatism (like e1) (wavelength), N(0.0.07).
            opt_c1 = rng_gaussian()             # Coma along y axis (wavelength), N(0.0.07).
            opt_c2 = rng_gaussian()             # Coma along x axis (wavelength), N(0.0.07).
            spher = rng_gaussian()              # Spherical aberration (wavelength), N(0.0.07).
            trefoil1 = rng_gaussian()           # Trefoil along y axis (wavelength), N(0.0.07).
            trefoil2 = rng_gaussian()           # Trefoil along x axis (wavelength), N(0.0.07).
            opt_obscuration = 0.1 + 0.4 * rng() # Linear dimension of central obscuration as fraction of pupil linear dimension, U(0.1, 0.5).
            lam_over_diam = .013 + 0.07 * rng() # Wavelength over diameter (arcsec), U(0.013, 0.083).
            
            psf_image = get_LSST_PSF(lam_over_diam, opt_defocus, 
                               opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                               atmos_fwhm, atmos_e, atmos_beta, spher, trefoil1, trefoil2, 0, 0,
                               fov_pixels, pixel_scale, upsample) 

        # Galaxy parameters .     
        gal_g = rng_gal_shear()             # Shear of the galaxy (magnitude of the shear in the "reduced shear" definition), U(0.01, 0.05).
        gal_beta = 2. * np.pi * rng()       # Shear position angle (radians), N(0,2*pi).
        gal_mu = 1 + rng() * 0.1            # Magnification, U(1.,1.1).
        theta = 2. * np.pi * rng()          # Rotation angle (radians), U(0,2*pi).
        dx = 2 * rng() - 1                  # Offset along x axis, U(-1,1).
        dy = 2 * rng() - 1                  # Offset along y axis, U(-1,1).
        gal_image = get_COSMOS_Galaxy(cosmos_catalog=cosmos_catalog, real_galaxy_catalog=real_galaxy_catalog, idx=idx,
                                      gal_g=gal_g, gal_beta=gal_beta,
                                      theta=theta, gal_mu=gal_mu, dx=dx, dy=dy,
                                      fov_pixels=fov_pixels, pixel_scale=pixel_scale, upsample=upsample)

        read_noise = 0.05 + 0.1 * rng()     # Standrad deviation of Gaussain read noise (ADU/pixel), U(0.05, 0.15).
        sky_level_pixel = 10 + rng() * 40    # Sky level (ADU/pixel), U(5,55).
        snr = rng_snr()
        gal_image_down = down_sample(gal_image.clone(), upsample) # Downsample galaxy image for SNR calculation.
        alpha = snr * torch.sqrt((sky_level_pixel+read_noise**2)/(gal_image_down**2).sum()) # Scale the flux of galaxy to meet SNR.
        gt = alpha * gal_image + sky_level_pixel 
        
        # Convolution using FFT.
        conv = ifftshift(ifft2(fft2(psf_image.clone()) * fft2(gt.clone()))).real
        
        # Downsample images to desired pixel scale.
        conv = down_sample(conv.clone(), upsample)
        psf = down_sample(psf_image.clone(), upsample)
        gt = down_sample(gt.clone(), upsample)

        # Add CCD noise (Poisson + Gaussian).
        conv = torch.max(torch.zeros_like(conv), conv) # Set negative pixels to zero.
        obs = torch.poisson(conv.clone()) + torch.normal(mean=torch.zeros_like(conv), std=read_noise*torch.ones_like(conv))
        obs = torch.max(torch.zeros_like(obs), obs) # Set negative pixels to zero.

        # Save images.
        torch.save(gt.clone(), os.path.join(data_path, 'gt', f"gt_{k}.pth"))
        torch.save(psf.clone(), os.path.join(data_path, 'psf', f"psf_{k}.pth"))
        torch.save(obs.clone(), os.path.join(data_path, 'obs', f"obs_{k}.pth"))

        if k >= n_train:
            # Simulate images with different SNR levels.
            for snr in snrs:

                alpha = snr * torch.sqrt((sky_level_pixel+read_noise**2)/(gal_image_down**2).sum()) # Scale the flux of galaxy to meet SNR
                gt_snr = alpha * gal_image + sky_level_pixel 
        
                # Convolution using FFT.
                conv_snr = ifftshift(ifft2(fft2(psf_image.clone()) * fft2(gt_snr.clone()))).real
                
                # Downsample images to desired pixel scale.
                conv_snr = down_sample(conv_snr.clone(), upsample)
                gt_snr = down_sample(gt_snr.clone(), upsample)
                
                # Add CCD noise (Poisson + Gaussian).
                conv_snr = torch.max(torch.zeros_like(conv_snr), conv_snr) # Set negative pixels to zero
                obs_snr = torch.poisson(conv_snr.clone()) + torch.normal(mean=torch.zeros_like(conv_snr), std=read_noise*torch.ones_like(conv_snr))
                obs_snr = torch.max(torch.zeros_like(obs_snr), obs_snr) # Set negative pixels to zero

                # Save Images.
                if not os.path.exists(os.path.join(data_path, f'gt_{snr}')):
                    os.mkdir(os.path.join(data_path, f'gt_{snr}'))
                if not os.path.exists(os.path.join(data_path, f'obs_{snr}')):
                    os.mkdir(os.path.join(data_path, f'obs_{snr}'))
                torch.save(gt_snr.clone(), os.path.join(data_path, f'gt_{snr}', f"gt_{k}.pth"))
                torch.save(obs_snr.clone(), os.path.join(data_path, f'obs_{snr}', f"obs_{k}.pth"))
                
            # Simulate PSF with shear error.
            for shear_err in shear_errs:
                g1_err = shear_err if rng() > 0.5 else -shear_err
                g2_err = shear_err if rng() > 0.5 else -shear_err
                psf_noisy = get_LSST_PSF(lam_over_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                                         atmos_fwhm, atmos_e, atmos_beta, spher, trefoil1, trefoil2, g1_err, g2_err,
                                         fov_pixels, pixel_scale, upsample)
                psf_noisy = down_sample(psf_noisy, upsample)
                # Save noisy PSFs.
                if not os.path.exists(os.path.join(data_path, f'psf_shear_err_{shear_err}')):
                    os.mkdir(os.path.join(data_path, f'psf_shear_err_{shear_err}'))
                torch.save(psf_noisy.clone(), os.path.join(data_path, f'psf_shear_err_{shear_err}', f"psf_{k}.pth"))
                    
            # Simulate PSF with FWHM error.
            for fwhm_err in fwhm_errs:
                fwhm = atmos_fwhm + fwhm_err if rng() > 0.5 else atmos_fwhm - fwhm_err
                fwhm = fwhm + 2*fwhm_err if fwhm < 0 else fwhm # Avoid negative FWHM.
                psf_noisy = get_LSST_PSF(lam_over_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                                         fwhm, atmos_e, atmos_beta, spher, trefoil1, trefoil2, 0, 0,
                                         fov_pixels, pixel_scale, upsample)
                psf_noisy = down_sample(psf_noisy, upsample)
                # Save noisy PSFs.
                if not os.path.exists(os.path.join(data_path, f'psf_fwhm_err_{fwhm_err}')):
                    os.mkdir(os.path.join(data_path, f'psf_fwhm_err_{fwhm_err}'))
                torch.save(psf_noisy.clone(), os.path.join(data_path, f'psf_fwhm_err_{fwhm_err}', f"psf_{k}.pth"))
                
        # Visualization
        if k < 50:
            gal_ori_image = real_galaxy_catalog.getGalImage(idx) # Read out original HST image for visualization
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.imshow(gal_ori_image.array, cmap='magma')
            plt.title('Original Galaxy')
            plt.subplot(2,2,2)
            plt.imshow(gt, cmap='magma')
            plt.title('Ground Truth')
            plt.subplot(2,2,3)
            plt.imshow(psf, cmap='magma')
            plt.title('PSF\n($FWHM={:.3f}$)'.format(atmos_fwhm) if survey=='LSST' else f'PSF: {psf_name}')
            plt.subplot(2,2,4)
            plt.imshow(obs, cmap='magma')
            plt.title('Observed Galaxy (SNR={:.1f})'.format(snr))
            plt.savefig(os.path.join(data_path, 'visualization', f"vis_{k}.jpg"), bbox_inches='tight')
            plt.close()
    


            
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for dataset.')
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=str, default='23.5', choices=['23.5', '25.2'])
    parser.add_argument('--fov_pixels', type=int, default=48)
    parser.add_argument('--pixel_scale', type=float, default=0.2)
    parser.add_argument('--upsample', type=int, default=4)
    opt = parser.parse_args()
    
    generate_data(data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new1/', train_split=0.7,
                  survey='LSST', I='23.5', fov_pixels=opt.fov_pixels, pixel_scale=opt.pixel_scale, upsample=opt.upsample,
                  snrs = [20, 40, 60, 80, 100, 150, 200, 300],
                  shear_errs=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
                  fwhm_errs=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3])
    