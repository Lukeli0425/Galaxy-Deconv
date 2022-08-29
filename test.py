import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
from torch.utils.data import DataLoader
from dataset import Galaxy_Dataset
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Richard_Lucy import Richard_Lucy
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR, estimate_shear, plot_psnr, plot_shear_err, plot_time_shear_err


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
            logging.raiseExceptions(f'Failed loading {model_file}!')

    def deconvolve(self, obs, psf):
        """Deconvolve PSF with unrolled ADMM model."""
        psf = torch.from_numpy(psf).unsqueeze(dim=0).unsqueeze(dim=0) if type(psf) is np.ndarray else psf
        obs = torch.from_numpy(obs).unsqueeze(dim=0).unsqueeze(dim=0) if type(obs) is np.ndarray else obs
        alpha = obs.ravel().mean()/0.33
        alpha = torch.Tensor(alpha.float()).view(1,1,1,1)

        output = self.model(obs.to(self.device), psf.to(self.device), alpha.to(self.device))
        rec = (output.cpu() * alpha.cpu()).squeeze(dim=0).squeeze(dim=0).numpy()
        
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
        logging.raiseExceptions('Failed loading pretrained model!')
    
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
            
            gt = gt.squeeze(dim=0).squeeze(dim=0).cpu()
            psf = psf.squeeze(dim=0).squeeze(dim=0).cpu()
            obs = obs.squeeze(dim=0).squeeze(dim=0).cpu()
            rec = rec.squeeze(dim=0).squeeze(dim=0).cpu()
        
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

# def test_shear(n_iters, llh, PnP, n_epochs, survey, I):
#     """Estimate shear with saved model."""
#     test_dataset = Galaxy_Dataset(train=False, survey=survey, I=I)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     gt_shear = []
#     obs_shear = []
#     rec_shear = []
#     fpfs_shear = []
    
#     result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
#     results_file = os.path.join(result_path, 'results.json')
#     if not os.path.exists(result_path):
#         os.mkdir(result_path)
#     try:
#         with open(results_file, 'r') as f:
#             results = json.load(f)
#         logging.info(f'Successfully loaded in {results_file}.')
#     except:
#         logging.warning(f'Failed loading in {results_file}.')
#         results = {}

#     model_file = f'saved_models/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs.pth'
#     model = ADMM_deconvolver(n_iters=n_iters, llh=llh, PnP=PnP, model_file=model_file)
    
#     for idx, ((obs, psf, M), gt) in enumerate(test_loader):
#         with torch.no_grad():
#             try:
#                 rec = io.imread(os.path.join(result_path, 'rec/', f"rec_{I}_{idx}.tiff"))
#             except:
#                 rec = model.deconvolve(obs, psf)
#             gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
#             psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
#             obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()

#             # Calculate shear
#             try:
#                 fpfs_shear.append(estimate_shear(obs, psf, use_psf=True))
#                 if abs(fpfs_shear[-1][0]) + abs(fpfs_shear[-1][1]) > 10:
#                     fpfs_shear.pop()
#                 else:
#                     gt_shear.append(estimate_shear(gt))
#                     obs_shear.append(estimate_shear(obs))
#                     rec_shear.append(estimate_shear(rec))
#                     logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})  fpfs:({:.3f},{:.3f})'.format(
#                         idx+1, len(test_loader),
#                         gt_shear[-1][0], gt_shear[-1][1],
#                         obs_shear[-1][0], obs_shear[-1][1],
#                         rec_shear[-1][0], rec_shear[-1][1],
#                         fpfs_shear[-1][0], fpfs_shear[-1][1]))
#             except:
#                 pass
#     gt_shear = np.array(gt_shear)
#     obs_shear = np.array(obs_shear)
#     rec_shear = np.array(rec_shear)
#     fpfs_shear = np.array(fpfs_shear)
#     obs_err_mean = np.mean(abs(obs_shear - gt_shear), axis=0)
#     rec_err_mean = np.mean(abs(rec_shear - gt_shear), axis=0)
#     fpfs_err_mean = np.mean(abs(fpfs_shear - gt_shear), axis=0)
#     obs_err_median = np.median(abs(obs_shear - gt_shear), axis=0)
#     rec_err_median = np.median(abs(rec_shear - gt_shear), axis=0)
#     fpfs_err_median = np.median(abs(fpfs_shear - gt_shear), axis=0)
#     obs_err_rms = np.sqrt(np.mean((obs_shear - gt_shear)**2, axis=0))
#     rec_err_rms = np.sqrt(np.mean((rec_shear - gt_shear)**2, axis=0))
#     fpfs_err_rms = np.sqrt(np.mean((fpfs_shear - gt_shear)**2, axis=0))
#     logging.info('Shear error (mean): ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
#                 obs_err_mean[0], obs_err_mean[1],
#                 rec_err_mean[0], rec_err_mean[1],
#                 fpfs_err_mean[0], fpfs_err_mean[1]))
#     logging.info('Shear error (median): ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
#                 obs_err_median[0], obs_err_median[1],
#                 rec_err_median[0], rec_err_median[1],
#                 fpfs_err_median[0], fpfs_err_median[1]))
#     logging.info('Shear error (RMS): ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
#                 obs_err_rms[0], obs_err_rms[1],
#                 rec_err_rms[0], rec_err_rms[1],
#                 fpfs_err_rms[0], fpfs_err_rms[1]))
    
#     # Save shear estimation
#     results['gt_shear'] = gt_shear.tolist()
#     results['obs_shear'] = obs_shear.tolist()
#     results['rec_shear'] = rec_shear.tolist()
#     results['fpfs_shear'] = fpfs_shear.tolist()
#     results['obs_err_mean'] = obs_err_mean.tolist()
#     results['rec_err_mean'] = rec_err_mean.tolist()
#     results['fpfs_err_mean'] = fpfs_err_mean.tolist()
#     results['obs_err_median'] = obs_err_median.tolist()
#     results['rec_err_median'] = rec_err_median.tolist()
#     results['fpfs_err_median'] = fpfs_err_median.tolist()
#     results['obs_err_rms'] = obs_err_rms.tolist()
#     results['rec_err_rms'] = rec_err_rms.tolist()
#     results['fpfs_err_rms'] = fpfs_err_rms.tolist()
#     with open(results_file, 'w') as f:
#         json.dump(results, f)
#     logging.info(f"Shear estimation results saved to {results_file}.")
#     return results

def test_shear(methods, n_iters, model_files, n_gal, snr):   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, 
                                  obs_folder=f'obs_{snr}', gt_folder=f'gt_{snr}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logging.info(f'Tesing method: {method}')
        result_path = os.path.join('results/', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, f'results.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
        if not str(snr) in results:
            results[str(snr)] = {}
        
        if n_iter > 0:
            if 'Richard-Lucy' in method:
                model = Richard_Lucy(n_iters=n_iter)
                model.to(device)
            else:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
                model.to(device)
                try: # Load the model
                    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                    logging.info(f'Successfully loaded in {model_file}.')
                except:
                    logging.raiseExceptions(f'Failed loading in {model_file} model!')   
            model.eval()     
    
        rec_shear = []
        for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
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
                    try:
                        rec_shear.append(estimate_shear(obs, psf, use_psf=True))
                    except:
                        rec_shear.append(obs_shear[idx])
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # Calculate shear
                    rec_shear.append(estimate_shear(rec))
                else:
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) #*= alpha.view(1,1,1)
                    rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    # Calculate shear
                    rec_shear.append(estimate_shear(rec))
            logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
                idx+1, len(test_loader),
                gt_shear[idx][0], gt_shear[idx][1],
                obs_shear[idx][0], obs_shear[idx][1],
                rec_shear[idx][0], rec_shear[idx][1]))
            if idx >= n_gal:
                break
        
        gt_shear, rec_shear = np.array(gt_shear), np.array(rec_shear)
        results[str(snr)]['rec_err_mean'] = np.mean(abs(rec_shear - gt_shear), axis=0).tolist()
        results[str(snr)]['rec_shear'] = rec_shear.tolist()
        results[str(snr)]['gt_shear'] = gt_shear.tolist()
        results[str(snr)]['obs_shear'] = obs_shear
        
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logging.info(f"Test results saved to {results_file}.")
    
    return results

def test_time(methods, n_iters, model_files, n_gal):  
    """Test the time of different models.""" 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        result_path = os.path.join('results', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, 'results.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
            
        if method == 'No_deconv':
            logging.info(f'Tested {method} on {n_gal} galaxies: Time = 0s')
            results['time'] = (0, n_gal)
            with open(results_file, 'w') as f:
                json.dump(results, f)
            logging.info(f"Test results saved to {results_file}.")  
            continue
        elif 'Richard-Lucy' in method:
            model = Richard_Lucy(n_iters=n_iter)
            model.to(device)
        else:
            model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
            model.to(device)
            try: # Load the model
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logging.info(f'Successfully loaded in {model_file}.')
            except:
                logging.raiseExceptions(f'Failed loading in {model_file} model!')   
        model.eval()     
        
        test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, psf_folder='psf/')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        time_start = time.time()
        for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
            with torch.no_grad():
                if 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    # rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                else:
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) #*= alpha.view(1,1,1)
                    # rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            if idx >= n_gal:
                break
        time_end = time.time()
        
        logging.info('Tested {} on {} galaxies: Time = {:.3f}s'.format(method, n_gal, time_end-time_start))
        results['time'] = (time_end-time_start, n_gal)
    
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logging.info(f"Test results saved to {results_file}.")  
    return results

if __name__ =="__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for tesing unrolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--llh', type=str, default='Poisson', choices=['Poisson', 'Gaussian'])
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    parser.add_argument('--n_gal', type=int, default=np.inf)
    parser.add_argument('--snr', type=int, default=20, choices=[20, 100])
    opt = parser.parse_args()
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    
    methods = ['No_deconv', 'Richard-Lucy(50)', 'Richard-Lucy(200)', 'Unrolled_ADMM(1)', 'Unrolled_ADMM(2)', 
               'Unrolled_ADMM(4)', 'Unrolled_ADMM(8)']
    n_iters = [0, 50, 200, 1, 2, 4, 8]
    model_files = [None, None, None,
                   "saved_models/Poisson_PnP_1iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_2iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_4iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_8iters_LSST23.5_50epochs.pth"]
    snrs = [20, 100]
    # test_time(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal)
    # test_shear(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, snr=20)
    # test_shear(methods=methods, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal, snr=100)
    # plot_psnr(n_iters=opt.n_iters, llh=opt.llh, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)
    # plot_shear_err(methods=methods)
    plot_time_shear_err(methods=methods, snrs=snrs)
