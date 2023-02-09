import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from models.Richard_Lucy import Richard_Lucy
from models.Tikhonet import Tikhonet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Wiener import Wiener
from utils.utils_data import get_dataloader
from utils.utils_ngmix import get_ngmix_Bootstrapper, make_data
from utils.utils_test import delta_2D, estimate_shear_new

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def test_psf_shear_err(methods, n_iters, model_files, n_gal, shear_err,
                       data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):
    logger = logging.getLogger('Noisy PSF Test (shear)')
    logger.info(' Running PSF shear_error=%s test with %s galaxies.\n', shear_err, n_gal)   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader(data_path=data_path, train=False, 
                                 psf_folder=f'psf_shear_err_{shear_err}/' if shear_err > 0 else 'psf/', 
                                 obs_folder='obs/', gt_folder='gt/')
    
    psf_delta = delta_2D(48, 48)
    
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(' Tesing method: %s', method)
        result_folder = os.path.join(result_path, method)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        results_file = os.path.join(result_folder, 'results_psf_shear_err.json')
        
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
        elif 'Tikhonet' in method or method == 'ShapeNet' or 'ADMM' in method:
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
                logger.info(' Successfully loaded in %s', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
            model.eval()    
    
        rec_shear = []
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    # obs = torch.max(torch.zeros_like(obs), obs)
                    gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    gt_shear.append(estimate_shear_new(gt, psf_delta))
                    rec_shear.append(estimate_shear_new(obs, psf_delta))
                elif method == 'FPFS':
                    # obs = torch.max(torch.zeros_like(obs), obs)
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
        
        # Save results to json file
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {}
        if not str(shear_err) in results:
            results[str(shear_err)] = {}
        results[str(shear_err)]['rec_shear'] = rec_shear
        if shear_err == 0:
            results[str(shear_err)]['gt_shear'] = gt_shear
        
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f" Test results saved to {results_file}.\n")
    
    return results
    
def test_psf_fwhm_err(method, n_iter, model_file, n_gal, fwhm_errs,
                      data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):
    logger = logging.getLogger('Noisy PSF Test (FWHM)')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psf_delta = delta_2D(48, 48)
    
    # for method, model_file, n_iter in zip(methods, model_files, n_iters):
    logger.info(' Tesing method: %s', method)
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results_psf_fwhm_err.json')
    
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
    elif 'Tikhonet' in method or method == 'ShapeNet' or 'ADMM' in method:
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
            logger.info(' Successfully loaded in %s', model_file)
        except:
            raise Exception('Failed loading in %s', model_file)
        model.eval()
    
    for fwhm_err in fwhm_errs:
        logger.info(' Running PSF fwhm_error=%s test with %s galaxies.\n', fwhm_err, n_gal)
        test_loader = get_dataloader(data_path=data_path, train=False,
                                     psf_folder=f'psf_fwhm_err_{fwhm_err}/' if fwhm_err > 0 else 'psf/',
                                     obs_folder='obs/', gt_folder='gt/')
        
        rec_shear, gt_shear = [], []
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    # obs = torch.max(torch.zeros_like(obs), obs)
                    gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    gt_shear.append(estimate_shear_new(gt, psf_delta))
                    rec_shear.append(estimate_shear_new(obs, psf_delta))
                elif method == 'FPFS':
                    # obs = torch.max(torch.zeros_like(obs), obs)
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
        
        # Save results to json file
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {}
        if not str(fwhm_err) in results:
            results[str(fwhm_err)] = {}
        results[str(fwhm_err)]['rec_shear'] = rec_shear
        if fwhm_err == 0:
            results[str(fwhm_err)]['gt_shear'] = gt_shear
        
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f" Test results saved to {results_file}.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for noisy PSF test.')
    parser.add_argument('--psf_error', type=str, default='shear', choices=['shear', 'fwhm'])
    parser.add_argument('--n_gal', type=int, default=10000)
    parser.add_argument('--result_path', type=str, default='results_200/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    # Uncomment the method to be tested.
    methods = {
        'No_Deconv': (0, None), 
        'FPFS': (0, None),
        # 'Wiener': (0, None), 
        'Richard-Lucy(10)': (10, None), 
        'Richard-Lucy(20)': (20, None), 
        'Richard-Lucy(30)': (30, None), 
        'Richard-Lucy(50)': (50, None), 
        'Richard-Lucy(100)': (100, None),
        'Tikhonet_Laplacian': (0, "saved_models_200/Tikhonet_Laplacian_MSE_20epochs.pth"), 
        'ShapeNet': (0, "saved_models_200/ShapeNet_Laplacian_50epochs.pth"), 
        # 'ADMMNet': (8, None),
        'Unrolled_ADMM_Gaussian(2)': (2, "saved_models_200/Gaussian_PnP_ADMM_2iters_MultiScale_20epochs.pth"), 
        'Unrolled_ADMM_Gaussian(4)': (4, "saved_models_200/Gaussian_PnP_ADMM_4iters_MultiScale_20epochs.pth"), 
        'Unrolled_ADMM_Gaussian(8)': (8, "saved_models_200/Gaussian_PnP_ADMM_8iters_MultiScale_20epochs.pth"),
        # 'Unrolled_ADMM_Gaussian(8)_MSE': (8, "saved_models_200/Gaussian_PnP_ADMM_8iters_MSE_20epochs.pth"),
        # 'Unrolled_ADMM_Gaussian(8)_Shape': (8, "saved_models_200/Gaussian_PnP_ADMM_8iters_Shape_20epochs.pth"),
        # 'Unrolled_ADMM_Gaussian(8)_No_SubNet': (8, "saved_models_200/Gaussian_PnP_ADMM_8iters_No_SubNet_MultiScale_20epochs.pth")
    }
    
    if opt.psf_error == 'shear':
        shear_errs = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        for method in methods:
            test_psf_shear_err(methods=method, n_iters=method[method][0], model_file=methods[method][1], n_gal=opt.n_gal, shear_errs=shear_errs,
                               data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)
    elif opt.psf_error == 'fwhm':
        fwhm_errs = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        for method in methods:
            test_psf_fwhm_err(method=method, n_iters=method[method][0], model_file=methods[method][1],  n_gal=opt.n_gal, fwhm_errs=fwhm_errs,
                              data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)
    else:
        raise ValueError('Invalid PSF test type.')
