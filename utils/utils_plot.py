import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def get_color(method):
    
    if 'Poisson' in method:
        color = 'xkcd:blue'
    elif 'Unrolled_ADMM' in method:
        color = 'xkcd:purple'
    elif 'ADMMNet' in method:
        color = 'xkcd:blue'
    elif 'Richard-Lucy' in method:
        color = 'xkcd:green' 
    elif 'Tikhonet' in method:
        color = 'xkcd:orange'
    elif method == 'ShapeNet':
        color = 'xkcd:pink'
    elif method == 'FPFS':
        color = 'xkcd:red'
    elif method == 'ngmix':
        color = 'xkcd:pink'
    elif method == 'No_Deconv':
        color = 'black'
    else:
        color = 'xkcd:brown'
        
    return color

def get_label(method):
    
    if 'Poisson' in method:
        label = 'Unrolled ADMM (Poisson)'
    elif 'Unrolled_ADMM' in method:
        label = 'Unrolled ADMM'
    elif 'Richard-Lucy' in method:
        label = 'Richardson-Lucy'
    elif method == 'Wiener':
        label = 'Wiener'
    elif 'Tikhonet' in method:
        label = 'Tikhonet'
    elif 'Identity' in method:
        label = 'Tikhonet (Identity)'
    elif method == 'ShapeNet':
        label = 'ShapeNet'
    elif method == 'FPFS':
        label = 'FPFS'
    elif method == 'ngmix':
        label = 'ngmix'
    elif method == 'No_Deconv':
        label = 'No Deconv'
    else:
        label = method
        
    return label

def plot_loss(train_loss, val_loss, epoch_min, model_save_path, model_name):
    n_epochs = len(train_loss)
    plt.figure(figsize=(12,7))
    plt.plot(range(1, n_epochs+1), train_loss, '-o', markersize=4, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_loss, '-o', markersize=4, label='Valid Loss')
    plt.plot([epoch_min+1], [val_loss[epoch_min]], 'ro', markersize=7, label='Best Epoch')
    plt.title(f'{model_name} Loss Curve', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.yscale("log")
    plt.legend(fontsize=15)
    file_name = f'./{model_save_path}/{model_name}_loss_curve.jpg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_psnr(n_iters, llh, PnP, n_epochs, survey, I):
    result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')
    except:
        logging.raiseExceptions(f'Failed loading in {results_file}.')

    # Plot PSNR distribution
    try:
        obs_psnr, rec_psnr = np.array(results['obs_psnr']), np.array(results['rec_psnr'])
        plt.figure(figsize=(12,10))
        plt.plot([10,35],[10,35],'r') # plot y=x line
        xy = np.vstack([obs_psnr, rec_psnr])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = obs_psnr[idx], rec_psnr[idx], z[idx]
        plt.scatter(x, y, c=z, s=8, cmap='Spectral_r')
        plt.colorbar()
        plt.title('PSNR of Test Results', fontsize=18)
        plt.xlabel('PSNR of Observed Galaxies', fontsize=15)
        plt.ylabel('PSNR of Recovered Galaxies', fontsize=15)
        plt.savefig(os.path.join(result_path, 'psnr.jpg'), bbox_inches='tight')
        plt.close()
    except:
        logging.warning('No PSNR data found!')

def plot_shear_err(n_iters, llh, PnP, n_epochs, survey, I):
    result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')
    except:
        logging.raiseExceptions(f'Failed loading in {results_file}.')

    # Plot shear error density distribution
    gt_shear = np.array(results['gt_shear'])
    obs_shear = np.array(results['obs_shear'])
    rec_shear = np.array(results['rec_shear'])
    fpfs_shear = np.array(results['fpfs_shear'])
    plt.figure(figsize=(15,4.2))
    plt.subplot(1,3,1)
    x = (obs_shear - gt_shear)[:,0]
    y = (obs_shear - gt_shear)[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    plt.xlabel('$e_1$', fontsize=13)
    plt.ylabel('$e_2$', fontsize=13)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.title('Observed Galaxy', fontsize=13)

    plt.subplot(1,3,2)
    x = (rec_shear - gt_shear)[:,0]
    y = (rec_shear - gt_shear)[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    plt.xlabel('$e_1$', fontsize=13)
    plt.ylabel('$e_2$', fontsize=13)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.title('Recovered Galaxy', fontsize=13)

    plt.subplot(1,3,3)
    x = (fpfs_shear - gt_shear)[:,0]
    y = (fpfs_shear - gt_shear)[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    plt.xlabel('$e_1$', fontsize=13)
    plt.ylabel('$e_2$', fontsize=13)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.title('Fourier Power Spectrum Deconvolution', fontsize=13)
    plt.savefig(os.path.join(result_path, 'shear_err.jpg'), bbox_inches='tight')

def plot_time_shear_err(methods, snrs):
    """Line plot of time and average shear error vs methods."""
    x = range(len(methods))
    colors = ['tab:blue', 'tab:purple']
    fig, ax1 = plt.subplots(figsize=(12,8))
    plt.xticks(x, methods, rotation=20, fontsize=11)
    for snr, color in zip(snrs, colors):
        shear_err_1, shear_err_2, t = [], [], []
        for method in methods:
            results_file = os.path.join('results', method, f'results.json')
            with open(results_file, 'r') as f:
                results = json.load(f)
            total_time, n_gal = results['time']
            shear_err_1.append(results[str(snr)]['rec_err_mean'][0])
            shear_err_2.append(results[str(snr)]['rec_err_mean'][1])
            t.append(total_time/n_gal)
    
        ax1.plot(x, shear_err_1, '--^', label=f'$g_1$ (SNR={snr})', color=color, markersize=8)
        ax1.plot(x, shear_err_2, '-.v', label=f'$g_2$ (SNR={snr})', color=color, markersize=8)
    ax1.set_ylabel('Average Shear Error', fontsize=15)
    ax1.set_ylim([0.01, 1])
    ax1.set_yscale('log')
    ax1.legend(loc="upper left", fontsize=12)
    
    ax2 = ax1.twinx()
    ax2.plot(x, t, '-o', label='Time per galaxy', color='tab:red', markersize=8)
    ax2.set_ylabel('Time/sec', fontsize=15)
    ax2.set_ylim([0, 0.05])
    ax2.legend(loc="upper right", fontsize=12)
    
    # ax1.xticks(x, methods, rotation=20, fontsize=14)
    ax1.set_xlabel('Methods', fontsize=15)
    plt.savefig(os.path.join('figures', 'time_shear_err.jpg'), bbox_inches='tight')
    plt.close()

def plot_shear_err_results(methods):
    """Draw line plot for systematic shear error in PSF vs shear estimation error."""
    color_list = ['tab:red', 'tab:olive', 'tab:purple', 'tab:blue', 'tab:cyan', 'tab:green', 'tab:orange']
    # Systematic shear error in PSF vs shear estimation error
    fig = plt.figure(figsize=(12,8))
    for method, color in zip(methods, color_list):
        result_path = os.path.join('results', method)
        results_file = os.path.join(result_path, 'results_psf_shear_err.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')

        shear_errs = results['shear_errs']
        rec_err_mean = np.array(results['rec_err_mean'])
        
        plt.plot(shear_errs, rec_err_mean[:,0], '-o', label='$g_1$, '+method, color=color)
        plt.plot(shear_errs, rec_err_mean[:,1], '--v', label='$g_2$, '+method, color=color)
        # plt.plot(shear_errs, rec_err_mean[:,1], '--v', label='$g_2$, '+method, color=color)
    
    plt.xlabel('Shear Error($\Delta_{g_1}$, $\Delta_{g_2}$) in PSF', fontsize=12)
    plt.ylabel('Average shear estimated error', fontsize=12)
    plt.xlim([-0.01, 0.41])
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join('figures', 'psf_shear_err.jpg'), bbox_inches='tight')
    plt.close()
    

def plot_seeing_err_results(methods):
    """Draw line plot for systematic seeing error in PSF vs shear estimation error."""
    color_list = ['tab:red', 'tab:olive', 'tab:purple', 'tab:blue', 'tab:cyan', 'tab:green', 'tab:orange']
    
    # Seeing error in PSF vs shear estimation error
    fig = plt.figure(figsize=(12,8))
    for method, color in zip(methods, color_list):
        result_path = os.path.join('results', method)
        results_file = os.path.join(result_path, 'results_psf_seeing_err.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')

        seeing_errs = results['seeing_errs']
        rec_err_mean = np.array(results['rec_err_mean'])
        
        plt.plot(seeing_errs, rec_err_mean[:,0], '-o', label='$g_1$, '+method, color=color)
        plt.plot(seeing_errs, rec_err_mean[:,1], '--v', label='$g_2$, '+method, color=color)
    
    plt.xlabel('Seeing Error in PSF (arcsec)', fontsize=12)
    plt.ylabel('Average shear estimated error', fontsize=12)
    plt.xlim([-0.01, 0.31])
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join('figures', 'psf_seeing_err.jpg'), bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    train_loss = [0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029,0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029,0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335]
    val_loss = [0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335]
    plot_loss(train_loss, val_loss, 10, './', 'Gaussian_ADMM')