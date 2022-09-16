from cmath import inf
import os
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Galaxy_Dataset
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Richard_Lucy import Richard_Lucy
from utils.utils import estimate_shear

def test_psf_shear_err(methods, n_iters, model_files, n_gal, shear_err):
    logger = logging.getLogger('PSF shear error test')
    logger.info(f'Running PSF shear_error={shear_err} test with {n_gal} galaxies.\n')   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, psf_folder=f'psf_shear_err{shear_err}/' if shear_err > 0 else 'psf/')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_path = os.path.join('results/', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, 'results_psf_shear_err.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
        if not str(shear_err) in results:
            results[str(shear_err)] = {}
        
        if n_iter > 0:
            if 'Richard-Lucy' in method:
                model = Richard_Lucy(n_iters=n_iter)
                model.to(device)
            elif 'ADMM' in method:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
                model.to(device)
                try: # Load the model
                    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                    logger.info(f'Successfully loaded in {model_file}.')
                except:
                    logger.error(f'Failed loading in {model_file} model!')   
            model.eval()     
    
        rec_shear = []
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_deconv':
                    gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    gt_shear.append(estimate_shear(gt))
                    obs_shear.append(estimate_shear(obs))
                    rec_shear.append(estimate_shear(obs))
                elif method == 'FPFS':
                    psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # try:
                    rec_shear.append(estimate_shear(obs, psf, use_psf=True))
                    # except:
                    #     rec_shear.append((gt_shear[idx][0],gt_shear[idx][1],gt_shear[idx][2]+1))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # Calculate shear
                    rec_shear.append(estimate_shear(rec))
                elif 'ADMM' in method:
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) #*= alpha.view(1,1,1)
                    rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # Calculate shear
                    rec_shear.append(estimate_shear(rec))
            # logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
            #     idx+1, len(test_loader),
            #     gt_shear[idx][0], gt_shear[idx][1],
            #     obs_shear[idx][0], obs_shear[idx][1],
            #     rec_shear[idx][0], rec_shear[idx][1]))
            if idx >= n_gal:
                break
        
        gt_shear, rec_shear = np.array(gt_shear), np.array(rec_shear)
        results[str(shear_err)]['rec_shear'] = rec_shear.tolist()
        results[str(shear_err)]['gt_shear'] = gt_shear.tolist()
        results[str(shear_err)]['obs_shear'] = obs_shear
        
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f"Test results saved to {results_file}.\n")
    
    return results
    
def test_psf_seeing_err(methods, n_iters, model_files, n_gal, seeing_err):
    logger = logging.getLogger('PSF shear seeing test')
    logger.info(f'Running PSF seeing_error={seeing_err} test with {n_gal} galaxies.\n')   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, psf_folder=f'psf_seeing_err{seeing_err}/' if seeing_err > 0 else 'psf/')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logger.info(f'Tesing method: {method}')
        result_path = os.path.join('results/', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, 'results_psf_seeing_err.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
        if not str(seeing_err) in results:
            results[str(seeing_err)] = {}
        
        if n_iter > 0:
            if 'Richard-Lucy' in method:
                model = Richard_Lucy(n_iters=n_iter)
                model.to(device)
            elif 'ADMM' in method:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
                model.to(device)
                try: # Load the model
                    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                    logger.info(f'Successfully loaded in {model_file}.')
                except:
                    logger.error(f'Failed loading in {model_file} model!')   
            model.eval()     
    
        rec_shear = []
        for (idx, ((obs, psf, alpha), gt)), _ in zip(enumerate(test_loader), tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_deconv':
                    gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    gt_shear.append(estimate_shear(gt))
                    obs_shear.append(estimate_shear(obs))
                    rec_shear.append(estimate_shear(obs))
                elif method == 'FPFS':
                    psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # try:
                    rec_shear.append(estimate_shear(obs, psf, use_psf=True))
                    # except:
                    #     rec_shear.append((gt_shear[idx][0],gt_shear[idx][1],gt_shear[idx][2]+1))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # Calculate shear
                    rec_shear.append(estimate_shear(rec))
                elif 'ADMM' in method:
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) #*= alpha.view(1,1,1)
                    rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # Calculate shear
                    rec_shear.append(estimate_shear(rec))
            # logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
            #     idx+1, len(test_loader),
            #     gt_shear[idx][0], gt_shear[idx][1],
            #     obs_shear[idx][0], obs_shear[idx][1],
            #     rec_shear[idx][0], rec_shear[idx][1]))
            if idx >= n_gal:
                break
        
        gt_shear, rec_shear = np.array(gt_shear), np.array(rec_shear)
        results[str(seeing_err)]['rec_shear'] = rec_shear.tolist()
        results[str(seeing_err)]['gt_shear'] = gt_shear.tolist()
        results[str(seeing_err)]['obs_shear'] = obs_shear
        
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(f"Test results saved to {results_file}.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for tesing unrolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=4)
    parser.add_argument('--llh', type=str, default='Poisson', choices=['Poisson', 'Gaussian'])
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    parser.add_argument('--n_gal', type=int, default=100)
    opt = parser.parse_args()
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    
    methods = ['No_deconv', 'FPFS',
               'Richard-Lucy(10)', 'Richard-Lucy(20)', 'Richard-Lucy(30)', 'Richard-Lucy(50)', 'Richard-Lucy(100)', 
               'Unrolled_ADMM(1)', 'Unrolled_ADMM(2)', 'Unrolled_ADMM(4)', 'Unrolled_ADMM(8)']
    n_iters = [0, 0, 10, 20, 30, 50, 100, 1, 2, 4, 8]
    model_files = [None, None, None, None, None, None, None,
                   "saved_models/Poisson_PnP_1iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_2iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_4iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_8iters_LSST23.5_50epochs.pth"]

    shear_errs=[0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for shear_err in shear_errs:
        test_psf_shear_err(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, shear_err=shear_err)
    
    seeing_errs=[0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    for seeing_err in seeing_errs:
        test_psf_seeing_err(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, seeing_err=seeing_err)
