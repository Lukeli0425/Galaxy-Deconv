import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from models.ADMMNet import ADMMNet
from models.Richard_Lucy import Richard_Lucy
from models.Tikhonet import Tikhonet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Wiener import Wiener
from score import score
from utils.utils_data import get_dataloader
from utils.utils_ngmix import get_ngmix_Bootstrapper, make_data
from utils.utils_test import delta_2D, estimate_shear_new


def test_shear(methods, n_iters, model_files, n_gal, snr,
               data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', result_path='results/'):
    logger = logging.getLogger('Shear Test')
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
        elif method == 'SCORE':
            g1 = score(gamma=1, n_starlet=4, n_shearlet=3, verbose=False, lip_eps=1e-3, 
                       n_itr=150, k=4, beta_factor=0.95, rip=True, tolerance=1e-6)
        elif method == 'Wiener':
            model = Wiener()
            model.to(device)
            model.eval()
        elif 'Richard-Lucy' in method:
            model = Richard_Lucy(n_iters=n_iter)
            model.to(device)
            model.eval()
        elif 'ADMMNet' in method:
            model = ADMMNet(n_iters=8, model_file="saved_models_abl/ResUNet_50epochs.pth", llh='Gaussian')
            model.to(device)
            model.eval()
            print("######")
        elif 'Tikhonet' in method or method == 'ShapeNet' or 'ADMM' in method:
            if method == 'Tikhonet':
                model = Tikhonet(filter='Identity')
            elif method == 'ShapeNet':
                model = Tikhonet(filter='Laplacian')
            elif 'Laplacian' in method:
                model = Tikhonet(filter='Laplacian')
            elif 'Gaussian' in method:
                model = Unrolled_ADMM(n_iters=n_iter,
                                      llh='Gaussian' if 'Gaussian' in method else 'Poisson', 
                                      SubNet='No_SubNet' not in method,
                                      PnP=True)
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
                elif method == 'SCORE':
                    # obs = torch.max(torch.zeros_like(obs), obs)
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    g1.deconvolve(obs=obs, psf=psf, gt=None)
                    rec = g1.solution
                    rec_shear.append(estimate_shear_new(obs, psf_delta))
                elif method == 'Wiener':
                    # obs = torch.max(torch.zeros_like(obs), obs)
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) 
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
            logger.info(f"Successfully loaded in {results_file}.")
        except:
            results = {} 
            logger.critical(f"Failed loading in {results_file}.")
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
    logger = logging.getLogger('Time Test')
    logger.info(f'Running time test with {n_gal} galaxies.\n')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader(data_path=data_path, train=False)
    
    psf_delta = delta_2D(48, 48)
    
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_folder = os.path.join(result_path, method)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        results_file = os.path.join(result_folder, 'results.json')

        if method == 'ngmix':
            boot = get_ngmix_Bootstrapper(psf_ngauss=1, ntry=2)
        elif method == 'SCORE':
            g1 = score(gamma=1, n_starlet=4, n_shearlet=3, verbose=False, lip_eps=1e-3, 
                       n_itr=150, k=4, beta_factor=0.95, rip=True, tolerance=1e-6)
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
        time_start = time.time()
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    # obs = torch.max(torch.zeros_like(obs), obs)
                    # gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(obs, psf_delta))
                elif method == 'FPFS':
                    # obs = torch.max(torch.zeros_like(obs), obs)
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear_new(obs, psf))
                # elif method == 'ngmix':
                #     obs = torch.max(torch.zeros_like(obs), obs)
                #     psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                #     obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                #     obs = make_data(obs_im=obs-obs.mean(), psf_im=psf)
                #     res = boot.go(obs)
                #     rec_shear.append((res['g'][0], res['g'][1], np.sqrt(res['g'][0]**2 + res['g'][1]**2)))
                elif method == 'Wiener':
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) 
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
                    
        time_end = time.time()
        
        logger.info('Tested {} on {} galaxies: Time = {:.3f}s.'.format(method, n_gal, time_end-time_start))
    
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Successfully loaded in {results_file}.")
        except:
            results = {} 
            logger.critical(f"Failed loading in {results_file}.")
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
    parser.add_argument('--result_path', type=str, default='results_200/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    methods = [
        # 'No_Deconv', 
        # 'FPFS', # 'SCORE', # 'ngmix', 
        # 'Wiener', 
        # 'Richard-Lucy(10)', 'Richard-Lucy(20)', 'Richard-Lucy(30)', 'Richard-Lucy(50)', 'Richard-Lucy(100)',
        'Tikhonet_Laplacian', 
        'ShapeNet', 
        # 'ADMMNet'
        # 'Unrolled_ADMM_Gaussian(2)', 'Unrolled_ADMM_Gaussian(4)', 'Unrolled_ADMM_Gaussian(8)'
    ]
    n_iters = [
        # 0,# 0, 0, 
        # 10, 20, 30, 50, 100,
        0, 0,
        # 2, 4, 8
    ]
    
    # model_files = [
    #     None, 
    #     None, 
    #     None, None, None, None, None, None,
    #     # "saved_models/Tikhonet_Laplacian_50epochs.pth",
    #     # "saved_models/ShapeNet_Laplacian_50epochs.pth",
    #     "saved_models/Gaussian_PnP_ADMM_2iters_MultiScale_20epochs.pth", 
    #     # "saved_models/Gaussian_PnP_ADMM_4iters_MultiScale_20epochs.pth",
    #     # "saved_models/Gaussian_PnP_ADMM_8iters_MultiScale_20epochs.pth",
    # ]
    
    model_files = [
        # None, 
        # None, 
        # None, None, None, None, None, 
        # None,
        "saved_models_200/Tikhonet_Laplacian_MSE_20epochs.pth",
        "saved_models_200/ShapeNet_Laplacian_30epochs.pth",
        # "saved_models_200/Gaussian_PnP_ADMM_2iters_MultiScale_20epochs.pth", 
        # "saved_models_200/Gaussian_PnP_ADMM_4iters_MultiScale_20epochs.pth",
        # "saved_models_200/Gaussian_PnP_ADMM_8iters_MultiScale_20epochs.pth",
    ]
    
    snrs = [20, 40, 60, 80, 100, 150, 200]

    # Ablation Test.
    # methods = [
    #     'Unrolled_ADMM_Gaussian(8)', 
    #     'Unrolled_ADMM_Gaussian(8)_MSE',
    #     'Unrolled_ADMM_Gaussian(8)_Shape',
    #     'Unrolled_ADMM_Gaussian(8)_No_SubNet'
    # ]
    # n_iters = [8, 8, 8, 8]
    # model_files = [
    #     "saved_models_200/Gaussian_PnP_ADMM_8iters_MultiScale_20epochs.pth",
    #     "saved_models_200/Gaussian_PnP_ADMM_8iters_MSE_20epochs.pth",
    #     "saved_models_200/Gaussian_PnP_ADMM_8iters_Shape_20epochs.pth",
    #     "saved_models_200/Gaussian_PnP_ADMM_8iters_No_SubNet_MultiScale_20epochs.pth",
    # ]
    
    
    for snr in snrs:
        test_shear(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, snr=snr,
                   data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)

    # for i in range(0, 3):
    #     test_time(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, 
    #             data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new3/', result_path=opt.result_path)
