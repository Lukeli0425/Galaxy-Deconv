import argparse
import logging
import os

import torch
from torch.optim import Adam

from models.Tikhonet import Tikhonet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.ResUNet import ResUNet
from utils.utils_data import get_dataloader
from utils.utils_plot import plot_loss
from utils.utils_torch import MultiScaleLoss, ShapeConstraint

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def train(model_name='Unrolled ADMM', n_iters=8, llh='Poisson', PnP=True, filter='Identity',
          n_epochs=10, lr=1e-4, loss='Default',
          data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5/', train_val_split=0.8, batch_size=32,
          model_save_path='./saved_models/', pretrained_epochs=0):

    logger = logging.getLogger('Train')
    train_loader, val_loader = get_dataloader(data_path=data_path, train=True, train_test_split=train_val_split, batch_size=batch_size)
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if 'ADMM' in model_name:
        logger.info(f'Start training Unrolled {"PnP-" if PnP else ""}ADMM with {llh} likelihood on {data_path} data for {n_epochs} epochs.')
        model = Unrolled_ADMM(n_iters=n_iters, llh=llh, PnP=PnP)
        model.to(device)
        if pretrained_epochs > 0:
            try:
                pretrained_file = os.path.join(model_save_path, f'{llh}{"_PnP" if PnP else ""}_{n_iters}iters_MSE_{pretrained_epochs}epochs.pth')
                model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
                logger.info(f'Successfully loaded in {pretrained_file}')
            except:
                logger.critical(f'Failed loading in {pretrained_file}')
        loss_fn = MultiScaleLoss()
    elif 'Tikhonet' in model_name:
        logger.info(f'Start training Tikhonet with {filter} filter on {data_path} data for {n_epochs} epochs.')
        model = Tikhonet(filter=filter)
        model.to(device)
        if pretrained_epochs > 0:
            try:
                pretrained_file = os.path.join(model_save_path, f'Tikhonet_{filter}_{pretrained_epochs}epochs.pth')
                model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
                logger.info(f'Successfully loaded in {pretrained_file}')
            except:
                logger.critical(f'Failed loading in {pretrained_file}')
        loss_fn = torch.nn.MSELoss()
    elif model_name == 'ShapeNet':
        logger.info(f'Start training ShapeNet on {data_path} data for {n_epochs} epochs.')
        model = Tikhonet(filter='Laplacian')
        model.to(device)
        if pretrained_epochs > 0:
            try:
                pretrained_file = os.path.join(model_save_path, f'ShapeNet_{pretrained_epochs}epochs.pth')
                model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
                logger.info(f'Successfully loaded in {pretrained_file}')
            except:
                logger.critical(f'Failed loading in {pretrained_file}')
        loss_fn = ShapeConstraint(device=device, fov_pixels=48, n_shearlet=2)
    elif model_name == 'ResUNet':
        logger.info(f'Start training ResUNet on {data_path} data for {n_epochs} epochs.')
        model = ResUNet()
        model.to(device)
        if pretrained_epochs > 0:
            try:
                pretrained_file = os.path.join(model_save_path, f'ResUNet_{pretrained_epochs}epochs.pth')
                model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
                logger.info(f'Successfully loaded in {pretrained_file}')
            except:
                logger.critical(f'Failed loading in {pretrained_file}')
        loss_fn = MultiScaleLoss()

    if loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss == 'MultiScale':
        loss_fn = MultiScaleLoss()
    elif loss == 'ShapeConstraint':
        loss_fn = ShapeConstraint(device=device, fov_pixels=48, n_shearlet=2)

    model_name = f'{llh}{"_PnP" if PnP else ""}_{loss if not loss=="Default" else ""}_{n_iters}iters' if 'ADMM' in model_name else (f'{model_name}_{filter}' if model_name=='Tikhonet' else model_name)
    
    optimizer = Adam(params=model.parameters(), lr = lr)

    train_loss_list = []
    val_loss_list = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for idx, ((obs, psf, alpha), gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
            # rec = model(obs/alpha) * alpha
            rec = model(obs, psf, alpha)
            loss = loss_fn(gt, rec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Evaluate on valid dataset
            if (idx+1) % 25 == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                        obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                        rec = model(obs, psf, alpha)
                        loss = loss_fn(gt, rec)
                        val_loss += loss.item()

                logger.info(" [{}: {}/{}]  train_loss={:.4f}  val_loss={:.4f}".format(
                                epoch+1, idx+1, len(train_loader),
                                train_loss/(idx+1),
                                val_loss/len(val_loader)))
    
        # Evaluate on train & valid dataset after every epoch
        train_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, ((obs, psf, alpha), gt) in enumerate(train_loader):
                obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
                loss = loss_fn(gt, rec)
                train_loss += loss.item()
            train_loss_list.append(train_loss/len(train_loader))
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
                loss = loss_fn(gt, rec)
                val_loss += loss.item()
            val_loss_list.append(val_loss/len(val_loader))

        logger.info(" [{}: {}/{}]  train_loss={:.4f}  val_loss={:.4f}".format(
                        epoch+1, len(train_loader), len(train_loader),
                        train_loss/(idx+1),
                        val_loss/len(val_loader)))

        if (epoch + 1) % 5 == 0:
            model_file_name = f'{model_name}_{epoch+1+pretrained_epochs}epochs.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, model_file_name))
            logger.info(f'Model saved to {os.path.join(model_save_path, model_file_name)}')

        # Plot loss curve
        plot_loss(train_loss_list, val_loss_list, model_save_path, model_name)

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--model', type=str, default='Unrolled ADMM', choices=['Unrolled ADMM', 'Tikhonet', 'ShapeNet', 'ResUNet'])
    parser.add_argument('--llh', type=str, default='Gaussian', choices=['Poisson', 'Gaussian'])
    parser.add_argument('--filter', type=str, default='Identity', choices=['Identity', 'Laplacian'])
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='MultiScale', choices=['Default', 'MultiScale', 'MSE', 'ShapeConstraint'])
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_epochs', type=int, default=0)
    opt = parser.parse_args()

    # from time import sleep
    # sleep(3600*2)

    train(model_name=opt.model, n_iters=opt.n_iters, llh=opt.llh, PnP=True, filter=opt.filter,
          n_epochs=opt.n_epochs, lr=opt.lr, loss=opt.loss,
          data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_new2/', train_val_split=opt.train_val_split, batch_size=opt.batch_size,
          model_save_path='./saved_models_abl/', pretrained_epochs=opt.pretrained_epochs)
