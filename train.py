import argparse
import logging
import os

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from models.ResUNet import ResUNet
from models.Tikhonet import Tikhonet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.unrolled_admm_gaussian import UnrolledADMMGaussian
from utils.utils_data import get_dataloader
from utils.utils_plot import plot_loss
from utils.utils_train import MultiScaleLoss, ShapeConstraint, get_model_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model_name='Unrolled ADMM', n_iters=8, llh='Poisson', PnP=True, remove_SubNet=False, filter='Laplacian',
          n_epochs=10, lr=1e-4, loss='MultiScale',
          data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', train_val_split=0.8, batch_size=32,
          model_save_path='./saved_models/', pretrained_epochs=0):
    
    model_name = get_model_name(method=model_name, loss=loss, filter=filter, n_iters=n_iters, llh=llh, PnP=PnP, remove_SubNet=remove_SubNet)
    logger = logging.getLogger('Train')
    logger.info(' Start training %s on %s data for %s epochs.', model_name, data_path, n_epochs)
    
    os.makedirs(model_save_path, exist_ok=True)
    
    train_loader, val_loader = get_dataloader(data_path=data_path, train=True, train_val_split=train_val_split, batch_size=batch_size, num_workers=12)
    
    if 'ADMM' in model_name:
        # model = Unrolled_ADMM(n_iters=n_iters, llh=llh, PnP=PnP, subnet=not remove_SubNet)
        model = UnrolledADMMGaussian(n_iters=n_iters, PnP=PnP, subnet=True, analysis=False)
    elif 'Tikhonet' in model_name:
        model = Tikhonet(filter=filter)
    elif 'ShapeNet' in model_name:
        model = Tikhonet(filter=filter)
    elif model_name == 'ResUNet':
        model = ResUNet()

    model.cuda()
    if pretrained_epochs > 0:
        try:
            pretrained_file = os.path.join(model_save_path, f'{model_name}_{pretrained_epochs}epochs.pth')
            model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
            logger.info(' Successfully loaded in %s.', pretrained_file)
        except:
            raise Exception(' Failed loading in %s!', pretrained_file)

    if 'ShapeNet' in model_name or loss == 'Shape':
        loss_fn = ShapeConstraint(device=device, fov_pixels=48, n_shearlet=2, gamma=1)
    elif loss == 'MSE':
        loss_fn = MSELoss().cuda()
    elif loss == 'MultiScale':
        loss_fn = MultiScaleLoss().cuda()
    
    optimizer = Adam(params=model.parameters(), lr = lr)

    train_loss_list, val_loss_list = [], []
    val_loss_min, epoch_min = 1.e9, 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for idx, ((obs, psf, alpha), gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, psf, alpha, gt = obs.cuda(), psf.cuda(), alpha.cuda(), gt.cuda()
            rec = model(obs, psf, alpha)
            loss = loss_fn(gt, rec)

            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            
            # Evaluate on valid dataset.
            if (idx+1) % 25 == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                        obs, psf, alpha, gt = obs.cuda(), psf.cuda(), alpha.cuda(), gt.cuda()
                        rec = model(obs, psf, alpha)
                        loss = loss_fn(gt, rec)
                        val_loss += loss.item()

                logger.info(" [{}: {}/{}]  train_loss={:.4g}  val_loss={:.4g}".format(
                                epoch+1, idx+1, len(train_loader),
                                train_loss, val_loss/len(val_loader)))
    
        # Evaluate on train & valid dataset after every epoch.
        train_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, ((obs, psf, alpha), gt) in enumerate(train_loader):
                obs, psf, alpha, gt = obs.cuda(), psf.cuda(), alpha.cuda(), gt.cuda()
                rec = model(obs, psf, alpha)
                loss = loss_fn(gt, rec)
                train_loss += loss.item()
            train_loss_list.append(train_loss/len(train_loader))
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                obs, psf, alpha, gt = obs.cuda(), psf.cuda(), alpha.cuda(), gt.cuda()
                rec = model(obs, psf, alpha)
                loss = loss_fn(gt, rec)
                val_loss += loss.item()
            val_loss_list.append(val_loss/len(val_loader))

        logger.info(" [{}: {}/{}]  train_loss={:.4g}  val_loss={:.4g}".format(
                        epoch+1, len(train_loader), len(train_loader),
                        train_loss/len(train_loader), val_loss/len(val_loader)))
        
        # Save model.
        if val_loss_min > val_loss or (epoch + 1) % 5 == 0:
            if val_loss_min > val_loss:
                val_loss_min = val_loss
                epoch_min = epoch
            model_file_name = f'{model_name}_{epoch+1+pretrained_epochs}epochs.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, model_file_name))
            logger.info(' Model saved to %s', os.path.join(model_save_path, model_file_name))

        # Plot loss curve.
        plot_loss(train_loss_list, val_loss_list, epoch_min, model_save_path, model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1', '2'])
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--model', type=str, default='Unrolled_ADMM', choices=['Unrolled_ADMM', 'Tikhonet', 'ShapeNet', 'ResUNet'])
    parser.add_argument('--llh', type=str, default='Gaussian', choices=['Gaussian', 'Poisson'])
    parser.add_argument('--remove_SubNet', action="store_true")
    parser.add_argument('--filter', type=str, default='Laplacian', choices=['Identity', 'Laplacian'])
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='MultiScale', choices=['MultiScale', 'MSE', 'Shape'])
    parser.add_argument('--train_val_split', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_epochs', type=int, default=0)
    args = parser.parse_args()

    # Select GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train(model_name=args.model, n_iters=args.n_iters, llh=args.llh, PnP=True, remove_SubNet=args.remove_SubNet, filter=args.filter,
          n_epochs=args.n_epochs, lr=args.lr, loss=args.loss,
          data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', train_val_split=args.train_val_split, batch_size=args.batch_size,
          model_save_path='./saved_models_test/', pretrained_epochs=args.pretrained_epochs)
