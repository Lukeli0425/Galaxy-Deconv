import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging
import argparse
import torch
from torch import nn
from torch.optim import Adam
from dataset import get_dataloader
from models.Unrolled_ADMM import Unrolled_ADMM
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import plot_loss

def train(n_iters=8, llh='Poisson', PnP=True, 
            n_epochs=10, lr=1e-4, survey='JWST', I=23.5, train_val_split=0.857, batch_size=32, 
            model_save_path='./saved_models/', load_pretrain=False,
            pretrained_file = None):

    logging.info(f'Start training unrolled {"PnP-" if PnP else ""}ADMM with {llh} likelihood on {survey}{I} data for {n_epochs} epochs.')
    train_loader, val_loader = get_dataloader(survey=survey, I=I, train_test_split=train_val_split, batch_size=batch_size)
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unrolled_ADMM(n_iters=n_iters, llh=llh, PnP=PnP)
    model.to(device)
    if load_pretrain:
        # try:
        model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
        logging.info(f'Successfully loaded in {pretrained_file}')
        # except:
        #     logging.critical(f'Failed loading in {pretrained_file}')

    optimizer = Adam(params=model.parameters(), lr = lr)
    loss_fn = MultiScaleLoss()

    train_loss_list = []
    val_loss_list = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for idx, ((obs, psf, alpha), gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
            rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
            # loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
            loss = loss_fn(gt, rec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Evaluate on valid dataset
            if (idx+1) % 20 == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                        obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                        rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
                        loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
                        val_loss += loss.item()

                logging.info(" [{}: {}/{}]  train_loss={:.4f}  val_loss={:.4f}".format(
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
                loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
                train_loss += loss.item()
            train_loss_list.append(train_loss/len(train_loader))
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
                loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
                val_loss += loss.item()
            val_loss_list.append(val_loss/len(val_loader))

        logging.info(" [{}: {}/{}]  train_loss={:.4f}  val_loss={:.4f}".format(
                        epoch+1, len(train_loader), len(train_loader),
                        train_loss/(idx+1),
                        val_loss/len(val_loader)))

        if (epoch + 1) % 5 == 0:
            model_file_name = f'{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{epoch+1}epochs.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, model_file_name))
            logging.info(f'P4IP model saved to {os.path.join(model_save_path, model_file_name)}')

        # Plot loss curve
        plot_loss(train_loss_list, val_loss_list, llh, PnP, n_iters, n_epochs, survey, I)

    return


if __name__ =="__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Arguments for training urolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--llh', type=str, default='Poisson', choices=['Poisson', 'Gaussian'])
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    parser.add_argument('--train_val_split', type=float, default=0.857)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_pretrain', action="store_true")
    opt = parser.parse_args()


    train(n_iters=opt.n_iters, llh=opt.llh, PnP=opt.PnP,
          n_epochs=opt.n_epochs, lr=opt.lr,
          survey=opt.survey, I=opt.I, train_val_split=opt.train_val_split, batch_size=opt.batch_size,
          load_pretrain=opt.load_pretrain,
          model_save_path='./saved_models/',
          pretrained_file='./saved_models/Poisson_PnP_8iters_LSST23.5_10epochs.pth')
