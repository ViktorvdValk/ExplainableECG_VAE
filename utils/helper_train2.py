"""
Helper module for ECG VAE training - contains Trainer_12 class.

This module provides the Trainer_12 class which handles:
- 12-lead ECG reconstruction using VAE
- Binary/multi-class prediction (LVF, MI)
- Adversarial training with contrastive loss

Usage:
    from utils.helper_train2 import Trainer_12
    
    trainer = Trainer_12(version='v1', batch_size=300, components=600,
                         beta_ls=0.0, beta_nls=0.0001, gamma=3, pred=1,
                         patience=15, lr=0.0001, labels=['lvf'], leads12=1,
                         binary=False, prev=True)
    trainer.train_12(num_epochs=500)
"""

import numpy as np
import gc
import wandb
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from itertools import cycle
from datetime import datetime
from utils.helper_loss import ResettableReduceLROnPlateau, SA_RPS_Loss
from vae.encoders import VAE_conv12, VAE_linear12, Adversarial_MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer_12:
    """Trainer class for 12-lead ECG VAE training with prediction head."""

    def __init__(self, version='v1', batch_size=32, learning_rate=0.001, pred=1, out_pred=2,
                 pca_init=False, cv=0, beta_ls=0.1, beta_nls=0.1, gamma=2, patience=15,
                 components=32, model_shape='conv', labels=['ncvd', 'pci_cbg'],
                 leads12=12, binary=False, prev=False, model_str='',
                 first_lead_only=False, all=False):
        """Initialize the Trainer_12 instance."""
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.version = version
        self.pca_init = pca_init
        self.cv = cv
        self.components = components
        self.beta_ls = beta_ls
        self.beta_nls = beta_nls
        self.patience = patience
        self.gamma = 0
        self.gamma_end = gamma
        self.pred = pred
        self.labels = labels
        self.leads12 = leads12
        self.binary = binary    
        self.first_lead_only = first_lead_only
        self.all = all
        
        if model_shape == 'linear':
            model = VAE_linear12(latent_dim=self.components, pred=pred).float()
        else:
            model = VAE_conv12(latent_dim=self.components, pred=pred, output_pred=out_pred).float()
            if prev and model_str:
                model.load_state_dict(torch.load(model_str))
        
        self.model = model.to(DEVICE)
        self.criterion_rec = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)     
        self.scheduler = ResettableReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.8, min_lr=1e-8)
        self.scaler = GradScaler()
        
        adv_mlp = Adversarial_MLP(latent_dim=self.components, pred2=pred, classes=out_pred).float()     
        self.adv_mlp = adv_mlp.to(DEVICE)
        self.optimizer_adv = torch.optim.Adam(
            self.adv_mlp.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scaler_adv = GradScaler()
        
        if binary:
            self.criterion_pred = nn.BCEWithLogitsLoss()
        else:        
            self.criterion_pred = SA_RPS_Loss(classes=4, combined=True)

    def run_epoch_12(self, loader, loader_p, train=True, epoch=0):
        """Run one training epoch for 12-lead ECG."""
        running_loss_recon = 0.0
        running_loss_pred = 0.0
        running_loss_adv = 0.0
        
        if epoch > 10:
            self.gamma = np.min([self.gamma_end, self.gamma + 0.0005])
        
        for recon_ecgs, pred_batch in zip(loader, cycle(loader_p)):
            recon_ecgs = recon_ecgs.reshape(-1, 12, 400)
            recon_ecgs = recon_ecgs.to(DEVICE).float()
            
            decoded, z_mean, z_log_var, _, _ = self.model(recon_ecgs)
            
            kl_div_ls = -0.5 * torch.sum(
                1 + z_log_var[:, :self.pred] - z_mean[:, :self.pred].pow(2) -
                torch.exp(z_log_var[:, :self.pred]), dim=1)
            kl_div_nls = -0.5 * torch.sum(
                1 + z_log_var[:, self.pred:] - z_mean[:, self.pred:].pow(2) -
                torch.exp(z_log_var[:, self.pred:]), dim=1)
            
            loss_rec12 = self.criterion_rec(
                recon_ecgs.reshape(recon_ecgs.shape[0], -1).float(),
                decoded.reshape(decoded.shape[0], -1).float())
            loss_recon1 = loss_rec12 + self.beta_ls * kl_div_ls.mean() + \
                         self.beta_nls * kl_div_nls.mean()
            running_loss_recon += loss_rec12.item()
            
            pred_ecgs, targets, _, _ = pred_batch
            pred_ecgs = pred_ecgs.reshape(-1, 12, 400)
            pred_ecgs = pred_ecgs.to(DEVICE).float()
            targets = targets.to(DEVICE)
            
            pred_decoded, z_mean, z_log_var, pred_train, pred_val = self.model(pred_ecgs)
            
            kl_div_ls = -0.5 * torch.sum(
                1 + z_log_var[:, :self.pred] - z_mean[:, :self.pred].pow(2) -
                torch.exp(z_log_var[:, :self.pred]), dim=1)
            kl_div_nls = -0.5 * torch.sum(
                1 + z_log_var[:, self.pred:] - z_mean[:, self.pred:].pow(2) -
                torch.exp(z_log_var[:, self.pred:]), dim=1)
            
            loss_rec12 = self.criterion_rec(
                pred_ecgs.reshape(pred_ecgs.shape[0], -1).float(),
                pred_decoded.reshape(pred_decoded.shape[0], -1).float())
            loss_recon2 = loss_rec12 + self.beta_ls * kl_div_ls.mean() + \
                         self.beta_nls * kl_div_nls.mean()
            running_loss_recon += loss_rec12.item()
            
            if train:
                loss_pred = self.criterion_pred(pred_train.squeeze(), targets.squeeze().float())
            else:
                loss_pred = self.criterion_pred(pred_val.squeeze(), targets.squeeze().float())
            running_loss_pred += loss_pred.item()
            
            loop = 5 if train else 1
            for _ in range(loop):
                pred_adv = self.adv_mlp.network(z_mean[:, self.pred:].detach())
                adv_loss = self.criterion_pred(pred_adv.squeeze(), targets.squeeze().float())
                
                if train:
                    self.optimizer_adv.zero_grad()
                    self.scaler_adv.scale(adv_loss).backward()
                    self.scaler_adv.step(self.optimizer_adv)
                    self.scaler_adv.update()
            
            pred_adv2 = self.adv_mlp(z_mean[:, self.pred:].detach())
            adv_loss2 = self.criterion_pred(pred_adv2.squeeze(), targets.squeeze().float())
            running_loss_adv += adv_loss2.item()
            
            loss = loss_recon1 + loss_recon2 + self.gamma * loss_pred + 5 * adv_loss2
            
            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        
        torch.cuda.empty_cache()
        gc.collect()
        return (running_loss_recon / len(loader), running_loss_pred / len(loader),
                running_loss_adv / len(loader), 0.0, 0.0)

    def train_12(self, num_epochs=10):
        """Main training method for 12-lead ECG."""
        print(f'Starting training for {self.leads12}-lead ECG')
        
        wandb.init(project="VAE_12", name=self.log_string(),
            config={"model": self.model._get_name, "cv": self.cv,
                    "components": self.components,
                    "beta_ls": self.beta_ls, "beta_nls": self.beta_nls,
                    "gamma": self.gamma, "leads12": self.leads12,
                    "labels": self.labels, "version": self.version})
        
        self.model.train()
        best_val_loss = np.inf
        patience_counter = 0
        
        train_loader = torch.load(
            f'data/train_loaderAS_{self.leads12}_{self.labels[0]}_{self.cv}'
            f'{self.binary}_{self.first_lead_only}_{self.all}all1of12.pt',
            weights_only=False)
        train_loader_p = torch.load(
            f'data/train_loaderAS_{self.leads12}_{self.labels[0]}_{self.cv}'
            f'{self.binary}_{self.first_lead_only}_{self.all}all1of12_p.pt',
            weights_only=False)
        val_loader = torch.load(
            f'data/val_loaderAS_{self.leads12}_{self.labels[0]}_{self.cv}'
            f'{self.binary}_{self.first_lead_only}_{self.all}all1of12.pt',
            weights_only=False)
        val_loader_p = torch.load(
            f'data/val_loaderAS_{self.leads12}_{self.labels[0]}_{self.cv}'
            f'{self.binary}_{self.first_lead_only}_{self.all}all1of12_p.pt',
            weights_only=False)
        
        for epoch in range(num_epochs):
            loss_recon, loss_pred, loss_adv, auc, auc2 = self.run_epoch_12(
                train_loader, train_loader_p, train=True, epoch=epoch)
            
            print(f'Train Epoch [{epoch+1}/{num_epochs}], '
                  f'Recon: {loss_recon:.4f}, Pred: {loss_pred:.4f}, '
                  f'Adv: {loss_adv:.4f}, AUC: {auc:.4f}, F1: {auc2:.4f}')
            
            wandb.log({"train_loss_recon": loss_recon, "train_loss_pred": loss_pred,
                      "train_loss_adv": loss_adv, "epoch": epoch})
            
            with torch.no_grad():
                self.model.eval()
                val_loss_recon, val_loss_pred, val_loss_adv, val_auc, val_f1 = \
                    self.run_epoch_12(val_loader, val_loader_p, train=False, epoch=epoch)
                
                print(f'Val Epoch [{epoch+1}/{num_epochs}], '
                      f'Recon: {val_loss_recon:.4f}, Pred: {val_loss_pred:.4f}, '
                      f'Adv: {val_loss_adv:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}')
                
                wandb.log({"val_loss_recon": val_loss_recon, "val_loss_pred": val_loss_pred,
                          "val_loss_adv": val_loss_adv, "epoch": epoch})
            
            if loss_pred < best_val_loss:
                best_val_loss = loss_pred
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > self.patience and epoch > 15:
                print('Early stopping!')
                break
        
        print('Finished Training')
        self.model.load_state_dict(best_state_dict)
        wandb.finish()

    def save_model(self):
        """Save model state dictionary to models/ directory."""
        torch.save(self.model.state_dict(), "models/" + self.log_string())
        torch.cuda.empty_cache()

    def log_string(self):
        """Generate model identifier string from configuration."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return (self.version + "_" + str(self.leads12) + "_ld_" + str(self.components) +
                "_beta_ls_" + str(self.beta_ls) + '_beta_nls_' + str(self.beta_nls) +
                "_gamma_" + str(self.gamma) + "_lr_" + str(self.learning_rate) + timestamp)

    def load_model(self, path):
        """Load model weights from saved checkpoint."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
