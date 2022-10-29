import os
import json
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import galsim
from utils.utils import PSNR


def get_LSST_PSF(lam, tel_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                  atmos_fwhm, atmos_e, atmos_beta, g1_err, g2_err,
                  fov_pixels, pixel_scale):
    """Simulate a PSF from a ground-based observation.

    Args:
        lam (float): Wavelength in nanometers.
        tel_diam (float): Diameter of the telescope in meters.
        opt_defocus (float): Defocus in units of incident light wavelength.
        opt_c1 (float): Coma along y in units of incident light wavelength.
        opt_c2 (float): Coma along x in units of incident light wavelength.
        opt_a1 (float): Astigmatism (like e2) in units of incident light wavelength. 
        opt_a2 (float): Astigmatism (like e1) in units of incident light wavelength. 
        opt_obscuration (float): Linear dimension of central obscuration as fraction of pupil linear dimension, [0., 1.).
        atmos_fwhm (float): The full width at half maximum of the Kolmogorov function for atmospheric PSF.
        atmos_e (float): Ellipticity of the shear to apply to the atmospheric component.
        atmos_beta (float): Position angle (in radians) of the shear to apply to the atmospheric component, twice the phase of a complex valued shear.
        g1_err (float): The first component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF.
        g2_err (float): The second component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF.
        fov_pixels (int): Width of the simulated images in pixels.
        pixel_scale (float): Pixel scale of the simulated image determining the resolution.

    Returns:
        torch.Tensor: Simulated PSF image with shape (fov_pixels, fov_pixels).
    """
    # Simulated PSF (optical + atmospheric)
    # Define the optical component of PSF
    lam_over_diam = lam * 1.e-9 / tel_diam # radians
    lam_over_diam *= 206265  # arcsec
    optics = galsim.OpticalPSF( lam_over_diam = lam_over_diam,
                                defocus = opt_defocus,
                                coma1 = opt_c1, coma2 = opt_c2,
                                astig1 = opt_a1, astig2 = opt_a2,
                                obscuration = opt_obscuration,
                                flux=1)
    
    # Define the atmospheric component of PSF
    atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1) # Note: the flux here is the default flux=1.
    atmos = atmos.shear(e=atmos_e, beta=atmos_beta*galsim.radians)
    psf = galsim.Convolve([atmos, optics], real_space=True)
    
    # add extra shear error
    psf = psf.shear(g1=g1_err, g2=g2_err)
    
    psf_image = galsim.ImageF(fov_pixels, fov_pixels)
    psf.drawImage(psf_image, scale=pixel_scale, method='auto')
    psf_image = torch.from_numpy(psf_image.array)
    psf_image = torch.max(torch.zeros_like(psf_image), psf_image)
            
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
                logging.info(f'Calculating Webb PSF: {instname} {filter}')
                psf_list = inst.calc_psf(fov_pixels=fov_pixels, oversample=1)
                psf = torch.from_numpy(psf_list[0].data)
                psf = torch.max(torch.zeros_like(psf), psf) # set negative pixels to zero
                psf /= psf.sum()
                psfs[instname+filter] = (psf, inst.pixelscale)
            except:
                pass
    psf_names = list(psfs.keys())

    return psf_names, psfs

def get_COSMOS_Galaxy(catalog, idx, gal_flux, sky_level, gal_e, gal_beta, theta, gal_mu, fov_pixels, pixel_scale, dx, dy):
    """Simulate a background galaxy with data from COSMOS real galaxy catalog.

    Args:
        catalog (galsim.RealGalaxyCatalog): A COSMOS Real Galaxy Catalog object, from which the galaxy are read out.
        idx (int): Index of the chosen galaxy in the catalog.
        gal_flux (float): Total flux of the galaxy in the simulated image.
        sky_level (float): Skylevel in the simulated image.
        gal_e (float): Ellipticity of the shear to apply.
        gal_beta (float): Position angle (in radians) of the shear to apply, twice the phase of a complex valued shear.
        theta (float): Rotation angle of the galaxy (in radians, positive means anticlockwise).
        gal_mu (float): The lensing magnification to apply.
        fov_pixels (int): Width of the simulated images in pixels.
        pixel_scale (float): Pixel scale of the simulated image determining the resolution.
        rng (galsim.UniformDeviate): A galsim random number generator object for simulation.

    Returns:
        torch.Tensor, numpy.array: Simulated galaxy image of shape (fov_pixels, fov_pixels) and original galaxy image in COSMOS dataset.
    """
    
    # Read out real galaxy from catalog
    gal_ori = galsim.RealGalaxy(catalog, index = idx, flux = gal_flux)
    psf_ori = catalog.getPSF(i=idx)
    gal_ori_image = catalog.getGalImage(idx)
    psf_ori_image = catalog.getPSFImage(idx)

    gal_ori = galsim.Convolve([psf_ori, gal_ori]) # concolve wth original PSF of HST
    gal = gal_ori.rotate(theta * galsim.radians) # Rotate by a random angle
    gal = gal.shear(e=gal_e, beta=gal_beta * galsim.radians) # Apply the desired shear
    gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.
    
    gal_image = galsim.ImageF(fov_pixels, fov_pixels)
    gal.drawImage(gal_image, scale=pixel_scale, offset=(dx,dy), method='auto')
    gal_image += sky_level * (pixel_scale**2)
    gal_image = torch.from_numpy(gal_image.array)
    gal_image = torch.max(torch.zeros_like(gal_image), gal_image)
    
    return gal_image, gal_ori_image.array


class Galaxy_Dataset(Dataset):
    """Simulated Galaxy Image Dataset inherited from torch.utils.data.Dataset."""
    def __init__(self, data_path='/mnt/WD6TB/tianaoli/dataset/', COSMOS_path='/mnt/WD6TB/tianaoli/', 
                 survey='LSST', I=23.5, fov_pixels=0, gal_max_shear=0.5, 
                 train=True, train_split=0.7, 
                 psf_folder='psf', obs_folder='obs', gt_folder='gt',
                 pixel_scale=0.2, atmos_max_shear=0.25, seeing=0.67):
        """Construction function for the PyTorch Galaxy Dataset.

        Args:
            survey (str): The telescope to simulate, 'LSST' or 'JWST'.
            I (float): A keyword argument that can be used to specify the sample to use, "23.5" or "25.2".
            fov_pixels (int): Width of the simulated images in pixels.
            gal_max_shear (float): Maximum shear applied to galaxies.
            train (bool): Whether the dataset is generated for training or testing.
            train_split (float, optional): Proportion of data used in train dataset, the rest will be used in test dataset. Defaults to 0.7.
            pixel_scale (float, optional): _description_. Defaults to 0.
            atmos_max_shear (float, optional): Maximum shear applied to atmospheric PSFs. Defaults to 0.25.
            seeing (float, optional): Average seeing. Defaults to 0.7.
            data_path (str, optional): Directory to save the dataset. Defaults to '/mnt/WD6TB/tianaoli/dataset/'.
            COSMOS_path (str, optional): Path to the COSMOS data. Defaults to '/mnt/WD6TB/tianaoli/'.
        """
        super(Galaxy_Dataset, self).__init__()
        # logging.info(f'Constructing {survey} Galaxy dataset.')
        
        # Initialize parameters
        self.train= train # Using train data or test data
        self.psf_folder = psf_folder # Path for PSFs
        self.obs_folder = obs_folder
        self.gt_folder = gt_folder
        self.COSMOS_dir = os.path.join(COSMOS_path, f"COSMOS_{I}_training_sample")
        self.train_split = train_split # n_train/n_total
        self.n_total = 0
        self.n_train = 0
        self.n_test = 0
        self.sequence = []
        self.info = {}

        self.survey = survey                # 'LSST' or 'JWST'
        self.I = I                          # I = 23.5 or 25.2 COSMOS data
        self.fov_pixels = 48 if self.survey=='LSST' else 64    # numbers of pixels in FOV
        self.pixel_scale = pixel_scale      # only used for LSST
        self.gal_max_shear = gal_max_shear
        self.atmos_max_shear = atmos_max_shear
        self.seeing = seeing
        
        # Create directory for the dataset
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.data_path = os.path.join(data_path, f'{self.survey}_{self.I}')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(os.path.join(self.data_path, 'obs')): # create directory for obs images
            os.mkdir(os.path.join(self.data_path, 'obs'))
        if not os.path.exists(os.path.join(self.data_path, 'gt')): # create directory for ground truth
            os.mkdir(os.path.join(self.data_path, 'gt'))
        if not os.path.exists(os.path.join(self.data_path, 'psf')): # create directory for PSF
            os.mkdir(os.path.join(self.data_path, 'psf'))
        if not os.path.exists(os.path.join(self.data_path, 'visualization')): # create directory for visualization
            os.mkdir(os.path.join(self.data_path, 'visualization'))

        # Read in real galaxy catalog
        try:
            self.real_galaxy_catalog = galsim.RealGalaxyCatalog(dir=self.COSMOS_dir, sample=str(self.I))
            self.n_total = self.real_galaxy_catalog.nobjects
            # logging.info(f'Successfully read in {self.n_total} real galaxies from {self.COSMOS_dir}.')
        except:
            logging.warning(f'Failed reading in real galaxies from {self.COSMOS_dir}.')

        # Read in information
        self.info_file = os.path.join(self.data_path, f'{self.survey}_{self.I}_info.json')
        try:
            # logging.info(f'Successfully loaded in {self.info_file}.')
            with open(self.info_file, 'r') as f:
                self.info = json.load(f)
            self.n_total = self.info['n_total']
            self.n_train = self.info['n_train']
            self.n_test = self.info['n_test']
            self.sequence = self.info['sequence']
        except:
            logging.critical(f'Failed reading information from {self.info_file}.')
            self.info = {'survey':survey, 'I':I, 'fov_pixels':fov_pixels, 'pixel_scale':pixel_scale, 
                         'gal_max_shear':gal_max_shear, 'atmos_max_shear':atmos_max_shear, 'seeing':seeing}      
            # Generate random sequence for data
            self.sequence = [i for i in range(self.n_total)]
            np.random.shuffle(self.sequence)
            self.n_train = int(self.train_split * self.n_total)
            self.info['n_total'] = self.n_total
            self.info['n_train'] = self.n_train
            self.info['n_test'] = self.real_galaxy_catalog.nobjects - self.n_train
            self.info['sequence'] = self.sequence
            with open(self.info_file, 'w') as f:
                json.dump(self.info, f)
            logging.info(f'Information saved in {self.info_file}.')

    def create_images(self, start_k=0, 
                      shear_errs=[0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                      seeing_errs=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]):
        
        logging.info(f'Simulating {self.survey} images.')
        
        random_seed = 345
        psnr_list = []
        
        # Calculate all Webb PSFs and split for train/test
        if self.survey == 'JWST':
            psf_names, psfs = get_Webb_PSF(fov_pixels=self.fov_pixels)
            np.random.shuffle(psf_names)
            train_psfs = psf_names[:int(len(psf_names) * self.train_split)]
            test_psfs = psf_names[int(len(psf_names) * self.train_split):]
        
        for k, _ in zip(range(start_k, self.n_total), tqdm(range(start_k, self.n_total))):
            idx = self.sequence[k] # index pf galaxy in the catalog
            rng = galsim.UniformDeviate(seed=random_seed+k+1) # Initialize the random number generator
            
            if self.survey == 'JWST': # Choose a Webb PSF 
                psf_name = np.random.choice(train_psfs) if k < self.n_train else np.random.choice(test_psfs)
                psf_image, pixel_scale = psfs[psf_name]
            elif self.survey == 'LSST': # Simulate a LSST PSF 
                # PSF parameters
                rng_gaussian = galsim.GaussianDeviate(seed=random_seed+k+10, mean=self.seeing, sigma=0.18)
                atmos_fwhm = 0 # arcsec (mean 0.7 for LSST)
                while atmos_fwhm < 0.35 or atmos_fwhm > 1.3: # sample fwhm
                    atmos_fwhm = rng_gaussian()
                atmos_e = rng() * self.atmos_max_shear # ellipticity of atmospheric PSF
                atmos_beta = 2. * np.pi * rng()     # radians
                atmos_shear = galsim.Shear(e=atmos_e, beta=atmos_beta * galsim.radians)
                opt_defocus = 0.3 + 0.4 * rng()     # wavelengths
                opt_a1 = 2*0.5*(rng() - 0.5)        # wavelengths (-0.29)
                opt_a2 = 2*0.5*(rng() - 0.5)        # wavelengths (0.12)
                opt_c1 = 2*1.*(rng() - 0.5)         # wavelengths (0.64)
                opt_c2 = 2*1.*(rng() - 0.5)         # wavelengths (-0.33)
                opt_obscuration = 0.165             # linear scale size of secondary mirror obscuration $(3.4/8.36)^2$
                lam = 700                           # nm    NB: don't use lambda - that's a reserved word.
                tel_diam = 8.36 # telescope diameter / meters (8.36 for LSST, 6.5 for JWST)
                pixel_scale = self.pixel_scale
                
                psf_image = get_LSST_PSF(lam, tel_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                                         atmos_fwhm, atmos_e, atmos_beta, 0, 0,
                                         self.fov_pixels, pixel_scale=pixel_scale) 
                if k >= self.n_train:
                    # Simulate PSF with shear error
                    for shear_err in shear_errs:
                        g1_err = shear_err if rng() > 0.5 else shear_err
                        g2_err = shear_err if rng() > 0.5 else shear_err
                        psf_noisy = get_LSST_PSF(lam, tel_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                                                 atmos_fwhm, atmos_e, atmos_beta, g1_err, g2_err,
                                                 self.fov_pixels, pixel_scale=pixel_scale)
                        # save PSF with error
                        if not os.path.exists(os.path.join(self.data_path, f'psf_shear_err{shear_err}')):
                            os.mkdir(os.path.join(self.data_path, f'psf_shear_err{shear_err}'))
                        torch.save(psf_noisy.clone(), os.path.join(self.data_path, f'psf_shear_err{shear_err}', f"psf_{self.I}_{k}.pth"))
                        
                    # Simulate PSF with seeing error
                    for seeing_err in seeing_errs:
                        # seeing_rng = galsim.GaussianDeviate(seed=random_seed+k+1, mean=0, sigma=seeing_err)
                        fwhm = atmos_fwhm + seeing_err if rng() > 0.5 else atmos_fwhm - seeing_err
                        fwhm = fwhm + 2*seeing_err if fwhm < 0 else fwhm
                        psf_noisy = get_LSST_PSF(lam, tel_diam, opt_defocus, opt_c1, opt_c2, opt_a1, opt_a2, opt_obscuration,
                                                 fwhm, atmos_e, atmos_beta, 0, 0, 
                                                 self.fov_pixels, pixel_scale=pixel_scale)
                        # save PSF with error
                        if not os.path.exists(os.path.join(self.data_path, f'psf_seeing_err{seeing_err}')):
                            os.mkdir(os.path.join(self.data_path, f'psf_seeing_err{seeing_err}'))
                        torch.save(psf_noisy.clone(), os.path.join(self.data_path, f'psf_seeing_err{seeing_err}', f"psf_{self.I}_{k}.pth"))
            
            # Galaxy parameters 
            sky_level = 1.e3                                        # ADU / arcsec^2
            gal_flux = 0.4e4 * np.exp(rng() * np.log(5e4/0.4e4))    # log uniform distribution
            gal_e = rng() * self.gal_max_shear  # shear of galaxy
            gal_beta = 2. * np.pi * rng()       # radians
            gal_shear = galsim.Shear(e=gal_e, beta=gal_beta*galsim.radians)
            gal_mu = 1 + rng() * 0.1            # mu = ((1-kappa)^2 - g1^2 - g2^2)^-1 (1.082)
            theta = 2. * np.pi * rng()          # radians
            dx = rng() - 0.5 # Offset by up to 1/2 pixel in each direction
            dy = rng() - 0.5
            gal_image, gal_orig = get_COSMOS_Galaxy(catalog=self.real_galaxy_catalog, idx=idx, 
                                                    gal_flux=gal_flux, sky_level=sky_level, 
                                                    gal_e=gal_e, gal_beta=gal_beta, 
                                                    theta=theta, gal_mu=gal_mu, dx=dx, dy=dy,
                                                    fov_pixels=self.fov_pixels, pixel_scale=pixel_scale)

            # Convolution via FFT
            conv = ifftshift(ifft2(fft2(psf_image) * fft2(gal_image))).real
            conv = torch.max(torch.zeros_like(conv), conv) # set negative pixels to zero

            # Add CCD noise (Poisson + Gaussian)
            obs = torch.poisson(conv) + torch.normal(mean=torch.zeros_like(conv), std=5*torch.ones_like(conv))
            obs = torch.max(torch.zeros_like(obs), obs) # set negative pixels to zero
            
            # Save images
            psnr = PSNR(obs, gal_image)
            psnr_list.append(psnr)
            torch.save(gal_image.clone(), os.path.join(self.data_path, 'gt', f"gt_{self.I}_{k}.pth"))
            torch.save(psf_image.clone(), os.path.join(self.data_path, 'psf', f"psf_{self.I}_{k}.pth"))
            torch.save(obs.clone(), os.path.join(self.data_path, 'obs', f"obs_{self.I}_{k}.pth"))
            # logging.info("Simulating Image:  [{:}/{:}]   PSNR={:.2f}".format(k+1, self.n_total, psnr))

            if k >= self.n_train:
                # Simulate different SNR
                for snr in [5, 10, 20, 40, 60, 80, 100, 150, 200]:
                    gal_flux = snr * (snr + np.sqrt((snr**2) + 4*sky_level*(self.fov_pixels**2)*(self.pixel_scale**2)))/2      
                    gal_image_snr, _ = get_COSMOS_Galaxy(catalog=self.real_galaxy_catalog, idx=idx, 
                                                    gal_flux=gal_flux, sky_level=sky_level, 
                                                    gal_e=gal_e, gal_beta=gal_beta, 
                                                    theta=theta, gal_mu=gal_mu, dx=dx, dy=dy,
                                                    fov_pixels=self.fov_pixels, pixel_scale=pixel_scale)

                    # Convolution via FFT
                    conv = ifftshift(ifft2(fft2(psf_image) * fft2(gal_image_snr))).real
                    conv = torch.max(torch.zeros_like(conv), conv) # set negative pixels to zero

                    # Add CCD noise (Poisson + Gaussian)
                    obs_snr = torch.poisson(conv) + torch.normal(mean=torch.zeros_like(conv), std=5*torch.ones_like(conv))
                    obs_snr = torch.max(torch.zeros_like(obs_snr), obs_snr) # set negative pixels to zero
                    
                    # Save
                    if not os.path.exists(os.path.join(self.data_path, f'gt_{snr}')):
                        os.mkdir(os.path.join(self.data_path, f'gt_{snr}'))
                    if not os.path.exists(os.path.join(self.data_path, f'obs_{snr}')):
                        os.mkdir(os.path.join(self.data_path, f'obs_{snr}'))
                    torch.save(gal_image_snr.clone(), os.path.join(self.data_path, f'gt_{snr}', f"gt_{self.I}_{k}.pth"))
                    torch.save(obs_snr.clone(), os.path.join(self.data_path, f'obs_{snr}', f"obs_{self.I}_{k}.pth"))
                    
            # Visualization
            if k < 50:
                plt.figure(figsize=(10,10))
                plt.subplot(2,2,1)
                plt.imshow(gal_orig)
                plt.title('Original Galaxy')
                plt.subplot(2,2,2)
                plt.imshow(gal_image)
                plt.title('Simulated Galaxy\n($g_1={:.3f}$, $g_2={:.3f}$)'.format(gal_shear.g1, gal_shear.g2))
                plt.subplot(2,2,3)
                plt.imshow(psf_image)
                plt.title('PSF\n($g_1={:.3f}$, $g_2={:.3f}$, FWHM={:.2f})'.format(atmos_shear.g1, atmos_shear.g2, atmos_fwhm) if self.survey=='LSST' else f'PSF: {psf_name}')
                plt.subplot(2,2,4)
                plt.imshow(obs)
                plt.title('Observed Galaxy\n($PSNR={:.2f}$)'.format(psnr))
                plt.savefig(os.path.join(self.data_path, 'visualization', f"{self.survey}_{self.I}_{k}.jpg"), bbox_inches='tight')
                plt.close()
        
        self.info['PSNR'] = psnr_list
        with open(self.info_file, 'w') as f:
            json.dump(self.info, f)

    def __len__(self):
        return self.n_train if self.train else self.n_test

    def __getitem__(self, i):
        idx = i if self.train else i + self.n_train
        
        psf_path = os.path.join(self.data_path, self.psf_folder)
        psf = torch.load(os.path.join(psf_path, f"psf_{self.I}_{idx}.pth")).unsqueeze(0)

        obs_path = os.path.join(self.data_path, self.obs_folder)
        obs = torch.load(os.path.join(obs_path, f"obs_{self.I}_{idx}.pth")).unsqueeze(0)
        obs = (obs - obs.min())/(obs.max() - obs.min())

        gt_path = os.path.join(self.data_path, self.gt_folder)
        gt = torch.load(os.path.join(gt_path, f"gt_{self.I}_{idx}.pth")).unsqueeze(0)
        gt = (gt - gt.min())/(gt.max() - gt.min())

        alpha = obs.ravel().mean().float()
        alpha = torch.Tensor(alpha).view(1,1,1)

        return (obs, psf, alpha), gt
            
            
def get_dataloader(survey='LSST', I=23.5, train_test_split=0.857, batch_size=32):
    """Generate dataloaders from Galaxy Dataset.

    Args:
        survey (str, optional): The survey of the dataset. Defaults to 'LSST'.
        I (float, optional): _description_. Defaults to 23.5.
        train_test_split (float, optional): Proportion of data used in train dataloader in train dataset, the rest will be used in valid dataloader. Defaults to 0.857.
        batch_size (int, optional): Batch size for training dataset. Defaults to 32.

    Returns:
        train_loader:  PyTorch data loader for train dataset.
        val_loader: PyTorch data loader for valid dataset.
    """
    train_dataset = Galaxy_Dataset(data_path='/mnt/WD6TB/tianaoli/dataset/', 
                                   COSMOS_path='/mnt/WD6TB/tianaoli/',
                                   survey=survey, I=I, train=True)
    
    train_size = int(train_test_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for dataset.')
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    opt = parser.parse_args()
    
    Dataset = Galaxy_Dataset(data_path='/mnt/WD6TB/tianaoli/dataset/', 
                             COSMOS_path='/mnt/WD6TB/tianaoli/',
                             survey=opt.survey, I=opt.I, pixel_scale=0.2)
    Dataset.create_images(start_k=0)
    