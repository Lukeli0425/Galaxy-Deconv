import argparse
import json
import logging
import os

import torch
from tqdm import tqdm

from models.Richard_Lucy import Richard_Lucy
from models.Tikhonet import Tikhonet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Wiener import Wiener
from utils.utils_data import get_dataloader
from utils.utils_test import delta_2D, estimate_shear

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def test_psf_shear_err(method, n_iters, model_file, n_gal, shear_errs,
                      data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path='results/'):
    logger = logging.getLogger('Noisy PSF Test (shear)')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psf_delta = delta_2D(48, 48)
    
    logger.info(' Tesing method: %s', method)
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results_psf_shear_err.json')
    
    # Load the model.
    model = None
    if method == 'Wiener':
        model = Wiener()
    elif 'Richard-Lucy' in method:
        model = Richard_Lucy(n_iters=n_iters)
    elif method == 'Tikhonet':
        model = Tikhonet(filter='Identity')
    elif method == 'ShapeNet' or 'Laplacian' in method:
        model = Tikhonet(filter='Laplacian')
    elif 'Gaussian' in method:
        model = Unrolled_ADMM(n_iters=n_iters, llh='Gaussian', PnP=True)
    else:
        model = Unrolled_ADMM(n_iters=n_iters, llh='Poisson', PnP=True)

    if model is not None:
        model.to(device)
        if 'Tikhonet' in method or 'ShapeNet' in method or 'ADMM' in method:
            try: # Load the pretrained wieghts.
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(' Successfully loaded in %s.', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
        model.eval()
    
    for shear_err in shear_errs:
        logger.info(' Running PSF shear_error=%s test with %s galaxies.\n', shear_err, n_gal)
        test_loader = get_dataloader(data_path=data_path, train=False,
                                     psf_folder=f'psf_shear_err_{shear_err}/' if shear_err > 0 else 'psf/')
        
        rec_shear, gt_shear = [], []
        for ((obs, psf, alpha), gt), _ in zip(test_loader, tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    gt_shear.append(estimate_shear(gt, psf_delta))
                    rec_shear.append(estimate_shear(obs, psf_delta))
                elif method == 'FPFS':
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(obs, psf))
                elif method == 'Wiener':
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha)
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
                else: # Unrolled ADMM, Wiener, Tikhonet, ShapeNet
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha)
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
        
        
        # Save results.
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(" Successfully loaded in %s.", results_file)
        except:
            results = {}
            logger.critical(" Failed loading in %s.", results_file)
            
        if not str(shear_err) in results:
            results[str(shear_err)] = {}
        results[str(shear_err)]['rec_shear'] = rec_shear
        if shear_err == 0:
            results[str(shear_err)]['gt_shear'] = gt_shear
        
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(" PSF test (shear) results saved to %s.\n", results_file)
    
    return results
    
def test_psf_fwhm_err(method, n_iters, model_file, n_gal, fwhm_errs,
                      data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path='results/'):
    logger = logging.getLogger('Noisy PSF Test (FWHM)')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psf_delta = delta_2D(48, 48)
    
    logger.info(' Tesing method: %s', method)
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results_psf_fwhm_err.json')
    
    # Load the model.
    model = None
    if method == 'Wiener':
        model = Wiener()
    elif 'Richard-Lucy' in method:
        model = Richard_Lucy(n_iters=n_iters)
    elif method == 'Tikhonet':
        model = Tikhonet(filter='Identity')
    elif method == 'ShapeNet' or 'Laplacian' in method:
        model = Tikhonet(filter='Laplacian')
    elif 'Gaussian' in method:
        model = Unrolled_ADMM(n_iters=n_iters, llh='Gaussian', PnP=True)
    else:
        model = Unrolled_ADMM(n_iters=n_iters, llh='Poisson', PnP=True)

    if model is not None:
        model.to(device)
        if 'Tikhonet' in method or 'ShapeNet' in method or 'ADMM' in method:
            try: # Load the pretrained wieghts.
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(' Successfully loaded in %s.', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
        model.eval()
    
    for fwhm_err in fwhm_errs:
        logger.info(' Running PSF fwhm_error=%s test with %s galaxies.\n', fwhm_err, n_gal)
        test_loader = get_dataloader(data_path=data_path, train=False,
                                     psf_folder=f'psf_fwhm_err_{fwhm_err}/' if fwhm_err > 0 else 'psf/')
        
        rec_shear, gt_shear = [], []
        for ((obs, psf, alpha), gt), _ in zip(test_loader, tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    gt_shear.append(estimate_shear(gt, psf_delta))
                    rec_shear.append(estimate_shear(obs, psf_delta))
                elif method == 'FPFS':
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(obs, psf))
                elif method == 'Wiener':
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha)
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
                else: # Unrolled ADMM, Wiener, Tikhonet, ShapeNet
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha)
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
        
        # Save results.
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(" Successfully loaded in %s.", results_file)
        except:
            results = {}
            logger.critical(" Failed loading in %s.", results_file)
            
        if not str(fwhm_err) in results:
            results[str(fwhm_err)] = {}
        results[str(fwhm_err)]['rec_shear'] = rec_shear
        if fwhm_err == 0:
            results[str(fwhm_err)]['gt_shear'] = gt_shear
        
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(" PSF test (FWHM) results saved to %s.\n", results_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for PSF robustness test.')
    parser.add_argument('--error', type=str, default='shear', choices=['shear', 'fwhm'])
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
    
    if opt.error == 'shear':
        shear_errs = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        for method, (n_iters, model_file) in methods.items():
            test_psf_shear_err(methods=method, n_iters=n_iters, model_file=model_file, n_gal=opt.n_gal, shear_errs=shear_errs,
                               data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)
    elif opt.error == 'fwhm':
        fwhm_errs = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        for method, (n_iters, model_file) in methods.items():
            test_psf_fwhm_err(method=method, n_iters=n_iters, model_file=model_file,  n_gal=opt.n_gal, fwhm_errs=fwhm_errs,
                              data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)
    else:
        raise ValueError('Invalid PSF robustness test type.')
