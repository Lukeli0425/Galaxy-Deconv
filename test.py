import os
import time
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.utils_data import get_dataloader
from models.Wiener import Wiener
from models.Richard_Lucy import Richard_Lucy
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Tikhonet import Tikhonet
from utils.utils_torch import MultiScaleLoss
from utils.utils_test import delta_2D, PSNR, estimate_shear_new
from utils.utils_ngmix import make_data, get_ngmix_Bootstrapper


class ADMM_deconvolver:
    """Wrapper class for unrolled ADMM deconvolution."""
    def __init__(self, n_iters=8, llh='Poisson', PnP=False, model_file=None):
        self.model_file = model_file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Unrolled_ADMM(n_iters=n_iters, llh=llh, PnP=PnP)
        self.model.to(self.device)
        # Load pretrained model
        try:
            self.model.load_state_dict(torch.load(model_file, map_location=torch.device(self.device)))
            logging.info(f'Successfully loaded in {model_file}.')
        except:
            logging.error(f'Failed loading {model_file}!')

    def deconvolve(self, obs, psf):
        """Deconvolve PSF with unrolled ADMM model."""
        psf = torch.from_numpy(psf).unsqueeze(dim=0).unsqueeze(dim=0) if type(psf) is np.ndarray else psf
        obs = torch.from_numpy(obs).unsqueeze(dim=0).unsqueeze(dim=0) if type(obs) is np.ndarray else obs
        alpha = obs.ravel().mean()/0.33
        alpha = torch.Tensor(alpha.float()).view(1,1,1,1)

        output = self.model(obs.to(self.device), psf.to(self.device), alpha.to(self.device))
        rec = (output.cpu() * alpha.cpu()).cpu().squeeze(dim=0).squeeze(dim=0).numpy()
        
        return rec

def test(n_iters, llh, PnP, n_epochs, survey, I):
    """Test the model."""     
    logging.info(f'Start testing unrolled {"PnP-" if PnP else ""}ADMM model with {llh} likelihood on {survey} data.')
    results = {} # dictionary to record the test results
    result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(os.path.join(result_path, 'rec')): # create directory for recovered image
        os.mkdir(os.path.join(result_path, 'rec'))
    if not os.path.exists(os.path.join(result_path, 'visualization')): # create directory for visualization
        os.mkdir(os.path.join(result_path, 'visualization'))
    
    test_dataset = Galaxy_Dataset(train=False, survey=survey, I=I)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unrolled_ADMM(n_iters=n_iters, llh=llh, PnP=PnP)
    model.to(device)
    
    # Load the model
    model_file = f'saved_models/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs.pth'
    try:
        model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
        logging.info(f'Successfully loaded in {model_file}.')
    except:
        logging.error('Failed loading pretrained model!')
    
    loss_fn = MultiScaleLoss()

    test_loss = 0.0
    obs_psnr = []
    rec_psnr = []
    model.eval()
    for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
        with torch.no_grad():
            obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
            rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
            loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
            test_loss += loss.item()
            rec *= alpha.view(1,1,1)
            
            gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).cpu()
            psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).cpu()
            obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).cpu()
            rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).cpu()
        
        # Save image
        # io.imsave(os.path.join(result_path, 'rec', f"rec_{I}_{idx}.tiff"), rec.numpy(), check_contrast=False)
        
        # Visualization
        if idx < 50:
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.imshow(gt)
            plt.title('Sheared Galaxy')
            plt.subplot(2,2,2)
            plt.imshow(psf)
            plt.title('PSF')
            plt.subplot(2,2,3)
            plt.imshow(obs)
            plt.title('Observed Galaxy\n($PSNR={:.2f}$)'.format(PSNR(gt, obs)))
            plt.subplot(2,2,4)
            plt.imshow(rec)
            plt.title('Recovered Galaxy\n($PSNR={:.2f}$)'.format(PSNR(gt, rec)))
            plt.savefig(os.path.join(result_path, 'visualization', f"vis_{I}_{idx}.jpg"), bbox_inches='tight')
            plt.close()
        else:
            break
        
        obs_psnr.append(PSNR(gt, obs))
        rec_psnr.append(PSNR(gt, rec))
        
        logging.info("Testing Image:  [{:}/{:}]  loss={:.4f}  PSNR: {:.2f} -> {:.2f}".format(
                        idx+1, len(test_loader), 
                        loss.item(), PSNR(gt, obs), PSNR(gt, rec)))
        
    logging.info("test_loss={:.4f}  PSNR: {:.2f} -> {:.2f}".format(
                    test_loss/len(test_loader),
                    np.mean(obs_psnr), np.mean(rec_psnr)))
        
    # Save results to json file
    results['test_loss'] = test_loss/len(test_loader)
    results['obs_psnr_mean'] = np.mean(obs_psnr)
    results['rec_psnr_mean'] = np.mean(rec_psnr)
    results['obs_psnr'] = obs_psnr
    results['rec_psnr'] = rec_psnr
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Test results saved to {results_file}.")

    return results


def test_shear(methods, n_iters, model_files, n_gal, snr,
               data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):
    logger = logging.getLogger('Shear test')
    logger.info(f'Running shear test with {n_gal} SNR={snr} galaxies.\n')   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader(data_path=data_path, train=False, 
                                 psf_folder='psf/', obs_folder=f'obs_{snr}/', gt_folder=f'gt_{snr}/')
    
    psf_delta = delta_2D(48, 48)
    
    gt_shear = []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_folder = os.path.join(result_path, method)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        results_file = os.path.join(result_folder, f'results.json')
        
        if method == 'ngmix':
            boot = get_ngmix_Bootstrapper(psf_ngauss=1, ntry=2)
        elif method == 'Wiener':
            model = Wiener()
            model.to(device)
            model.eval()
        elif 'Richard-Lucy' in method:
            model = Richard_Lucy(n_iters=n_iter)
            model.to(device)
            model.eval()
        elif method == 'Tikhonet' or method == 'ShapeNet' or 'ADMM' in method:
            if method == 'Tikhonet':
                model = Tikhonet(filter='Identity')
            elif method == 'ShapeNet':
                model = Tikhonet(filter='Laplacian')
            elif 'Laplacian' in method:
                model = Tikhonet(filter='Laplacian')
            elif 'Gaussian' in method:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Gaussian', PnP=True)
            else:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
            model.to(device)
            try: # Load the model
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(f'Successfully loaded in {model_file}.')
            except:
                logger.error(f'Failed loading in {model_file}.')
            model.eval()
    
        rec_shear = []
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    obs = torch.max(torch.zeros_like(obs), obs)
                    gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    gt_shear.append(estimate_shear_new(gt, psf_delta))
                    rec_shear.append(estimate_shear_new(obs, psf_delta))
                elif method == 'FPFS':
                    obs = torch.max(torch.zeros_like(obs), obs)
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(obs, psf))
                elif method == 'ngmix':
                    obs = torch.max(torch.zeros_like(obs), obs)
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = make_data(obs_im=obs-obs.mean(), psf_im=psf)
                    res = boot.go(obs)
                    rec_shear.append((res['g'][0], res['g'][1], np.sqrt(res['g'][0]**2 + res['g'][1]**2)))
                elif method == 'Wiener':
                    obs = torch.max(torch.zeros_like(obs), obs)
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf, snr) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(rec, psf_delta))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(rec, psf_delta))
                else: # ADMM, Tikhonet
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha)
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(rec, psf_delta))
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
        if not str(snr) in results:
            results[str(snr)] = {}
        results[str(snr)]['rec_shear'] = rec_shear
        results[str(snr)]['gt_shear'] = gt_shear
        
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f"Shear test results saved to {results_file}.\n")
    
    return results

def test_time(methods, n_iters, model_files, n_gal,
              data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):  
    """Test the time of different models."""
    logger = logging.getLogger('time test')
    logger.info(f'Running time test with {n_gal} galaxies.\n')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    psf_delta = delta_2D(48, 48)
    
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_folder = os.path.join(result_path, method)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        results_file = os.path.join(result_folder, 'results.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
        
        if method == 'ngmix':
            boot = get_ngmix_Bootstrapper(psf_ngauss=1, ntry=2)
        elif method == 'Wiener':
            model = Wiener()
            model.to(device) 
            model.eval()
        elif 'Richard-Lucy' in method:
            model = Richard_Lucy(n_iters=n_iter)
            model.to(device)
            model.eval() 
        else:
            if method == 'Tikhonet':
                model = Tikhonet()
            elif 'Gaussian' in method:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Gaussian', PnP=True)
            else:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
            model.to(device)
            try: # Load the model
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(f'Successfully loaded in {model_file}.')
            except:
                logger.error(f'Failed loading in {model_file}.')
            model.eval()    
        
        test_loader = get_dataloader(data_path=data_path, train=False)

        rec_shear = []
        time_start = time.time()
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    # gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(obs, psf_delta))
                elif method == 'FPFS':
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(obs, psf))
                elif method == 'Wiener':
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf, 20) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(rec, psf_delta))
                elif method == 'ngmix':
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = make_data(obs_im=obs, psf_im=psf)
                    res = boot.go(obs)
                    rec_shear.append((res['g'][0], res['g'][1], np.sqrt(res['g'][0]**2 + res['g'][1]**2)))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(rec, psf_delta))
                else: # ADMM, Tikhonet
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(rec, psf_delta))
                    
        time_end = time.time()
        
        logger.info('Tested {} on {} galaxies: Time = {:.3f}s.'.format(method, n_gal, time_end-time_start))
        results['time'] = (time_end-time_start, n_gal)
    
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f"Time test results saved to {results_file}.\n")  
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for testing.')
    parser.add_argument('--n_gal', type=int, default=10000)
    parser.add_argument('--result_path', type=str, default='results4/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    methods = [
        'No_Deconv', 
        # 'FPFS', 'ngmix', 
        # 'Richard-Lucy(10)', 'Richard-Lucy(20)', 'Richard-Lucy(30)', 'Richard-Lucy(50)', 'Richard-Lucy(100)', 
        'Tikhonet', 'ShapeNet', # 'Tikhonet_Laplacian',
        # 'Unrolled_ADMM(2)', 'Unrolled_ADMM(4)', 'Unrolled_ADMM(8)', 'Unrolled_ADMM(6)',
        # 'Unrolled_ADMM_Gaussian(6)',
        'Unrolled_ADMM_Gaussian(2)', 'Unrolled_ADMM_Gaussian(4)', 'Unrolled_ADMM_Gaussian(8)'
    ]
    n_iters = [
        0, 
        # 0, 0, 
        # 10, 20, 30, 50, 100,
        0, 0, # 0,
        # 2, 4, 6, 8, 
        2, 4, 8, 
    ]
    model_files = [
        None,
        # None, None,
        # None, None, None, None, None,
        "saved_models3/Tikhonet_Identity_20epochs.pth",
        "saved_models3/ShapeNet_15epochs.pth",
        # "saved_models2/Tikhonet_Laplacian_50epochs.pth",
        # "saved_models2/Poisson_PnP_1iters_50epochs.pth",
        # "saved_models2/Poisson_PnP_2iters_50epochs.pth",
        # "saved_models2/Poisson_PnP_4iters_50epochs.pth",
        # "saved_models2/Poisson_PnP_6iters_50epochs.pth",
        # "saved_models2/Poisson_PnP_8iters_50epochs.pth",
        # "saved_models3/Gaussian_PnP_1iters_20epochs.pth",
        "saved_models3/Gaussian_PnP_2iters_25epochs.pth",
        "saved_models3/Gaussian_PnP_4iters_20epochs.pth",
        # "saved_models3/Gaussian_PnP_6iters_20epochs.pth",
        "saved_models3/Gaussian_PnP_8iters_15epochs.pth"
    ]
    
    snrs = [300]

    for snr in snrs:
        test_shear(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, snr=snr,
                   data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new2/', result_path=opt.result_path)

    # test_time(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal,
    #           data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new2/', result_path=opt.result_path)
