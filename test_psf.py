import os
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import torch
from utils.utils_data import get_dataloader
from models.Wiener import Wiener
from models.Richard_Lucy import Richard_Lucy
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Tikhonet import Tikhonet
from utils.utils_test import delta_2D, estimate_shear_new
from utils.utils_ngmix import make_data, get_ngmix_Bootstrapper

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def test_psf_shear_err(methods, n_iters, model_files, n_gal, shear_err,
                       data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):
    logger = logging.getLogger('Noisy PSF Test (shear)')
    logger.info(f'Running PSF shear_error={shear_err} test with {n_gal} galaxies.\n')   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader(data_path=data_path, train=False, 
                                 psf_folder=f'psf_shear_err_{shear_err}/' if shear_err > 0 else 'psf/', obs_folder=f'obs/', gt_folder=f'gt/')
    
    psf_delta = delta_2D(48, 48)
    
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_folder = os.path.join(result_path, method)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        results_file = os.path.join(result_folder, f'results_psf_shear_err.json')
        
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
                logger.info(f'Successfully loaded in {model_file}.')
            except:
                logger.exception(f'Failed loading in {model_file}.')
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
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
        if not str(shear_err) in results:
            results[str(shear_err)] = {}
        results[str(shear_err)]['rec_shear'] = rec_shear
        results[str(shear_err)]['gt_shear'] = gt_shear
        
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f"Test results saved to {results_file}.\n")
    
    return results
    
def test_psf_fwhm_err(methods, n_iters, model_files, n_gal, fwhm_err,
                      data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):
    logger = logging.getLogger('Noisy PSF Test (FWHM)')
    logger.info(f'Running PSF fwhm_error={fwhm_err} test with {n_gal} galaxies.\n')   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(data_path=data_path, train=False, 
                                 psf_folder=f'psf_fwhm_err_{fwhm_err}/' if fwhm_err > 0 else 'psf/', obs_folder=f'obs/', gt_folder=f'gt/')
    
    psf_delta = delta_2D(48, 48)
    
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_folder = os.path.join(result_path, method)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        results_file = os.path.join(result_folder, f'results_psf_fwhm_err.json')
        
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
                logger.info(f'Successfully loaded in {model_file}.')
            except:
                logger.error(f'Failed loading in {model_file}.')
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
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {}
        if not str(fwhm_err) in results:
            results[str(fwhm_err)] = {}
        results[str(fwhm_err)]['rec_shear'] = rec_shear
        results[str(fwhm_err)]['gt_shear'] = gt_shear
        
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f"Test results saved to {results_file}.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for noisy PSF test.')
    parser.add_argument('--n_gal', type=int, default=10000)
    parser.add_argument('--result_path', type=str, default='results/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    methods = [
        'No_Deconv', 
        'FPFS', # 'SCORE', # 'ngmix', 
        # 'Wiener', 'Richard-Lucy(10)', 'Richard-Lucy(20)', 'Richard-Lucy(30)', 'Richard-Lucy(50)', 
        'Richard-Lucy(100)',
        'Tikhonet_Laplacian', 'ShapeNet', 
        'Unrolled_ADMM_Gaussian(2)', 'Unrolled_ADMM_Gaussian(4)', 'Unrolled_ADMM_Gaussian(8)'
    ]
    n_iters = [
        0, 
        0,# 0, 0, 
        # 0, 10, 20, 30, 50, 100,
        100,
        0, 0,
        2, 4, 8
    ]
    
    # model_files = [
    #     None, 
    #     None, 
    #     # None, None, None, None, None, 
    #     None,
    #     "saved_models_200/Tikhonet_Laplacian_MSE_30epochs.pth",
    #     "saved_models_200/ShapeNet_Laplacian_50epochs.pth",
    #     "saved_models_200/Gaussian_PnP_ADMM_2iters_MultiScale_20epochs.pth", 
    #     "saved_models_200/Gaussian_PnP_ADMM_4iters_MultiScale_20epochs.pth",
    #     "saved_models_200/Gaussian_PnP_ADMM_8iters_MultiScale_20epochs.pth",
    # ]
    
    
    model_files = [
        None, 
        None, 
        # None, None, None, None, None, 
        None,
        "saved_models/Tikhonet_Laplacian_50epochs.pth",
        "saved_models/ShapeNet_Laplacian_50epochs.pth",
        "saved_models/Gaussian_PnP_ADMM_2iters_MultiScale_50epochs.pth", 
        "saved_models/Gaussian_PnP_ADMM_4iters_MultiScale_50epochs.pth",
        "saved_models/Gaussian_PnP_ADMM_8iters_MultiScale_50epochs.pth",
    ]
    
    
    shear_errs = [0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    shear_errs = [0, 0.001, 0.002, 0.003, 0.005, 0.007]
    for shear_err in shear_errs:
        test_psf_shear_err(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, shear_err=shear_err,
                           data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new2/', result_path=opt.result_path)
    
    # fwhm_errs = [0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    # fwhm_errs = [0, 0.001, 0.002, 0.003, 0.005, 0.007]
    # for fwhm_err in fwhm_errs:
    #     test_psf_fwhm_err(methods=methods, n_iters=n_iters, model_files=model_files,  n_gal=opt.n_gal, fwhm_err=fwhm_err,
    #                        data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new2/', result_path=opt.result_path)
