"""VAE model definitions for 12-lead ECG.

    - GradientReversalFunction / GradientReversalLayer: domain-adversarial
      gradient reversal (identity forward, negated gradient backward)
    - Adversarial_MLP: classifier that debiases the non-label latent space
    - VAE_linear12: fully-connected VAE with a linear label regressor
    - VAE_conv12: 2D-residual conv VAE for the 12-lead layout
    - VAE_linear1to12: cross-lead model (1-lead <-> 12-lead generation)
"""

import torch
import torch.nn as nn
import numpy as np
from utils.encoder_utils import LinearResidualBlock, Reshape, GradientReversal, ResidualBlock, ResidualBlockTranspose

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_sig_dim(k, p, n, input_dim):
    sig_dim = np.ceil((input_dim - 3 * (k - 1) + 2 * p) / 2)
    for _ in range(n + 1):
        sig_dim -= k - 1 - 2 * p
        sig_dim = sig_dim / 2
        sig_dim = np.ceil(sig_dim)

    return int(sig_dim)

class GradientReversalFunction(torch.autograd.Function):
    """autograd: forward = identity clone, backward = negated gradient."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output  # Negates the gradients

class GradientReversalLayer(nn.Module):
    """nn.Module wrapper around GradientReversalFunction."""

    def forward(self, x):
        return GradientReversalFunction.apply(x)

# create adversarial network to debias latent space
class Adversarial_MLP(nn.Module):
    """Adversarial MLP: *attempts* to predict the class label from the
    (non-label) part of the latent code.

    Because its gradients pass through a gradient-reversal layer,
    back-prop into the encoder actively strips label information from
    the latent space (a debiasing / fairness regulariser).
    """
    def __init__(self, latent_dim=600, pred2=1, classes=2, three_class=False, AS=False, lead1=False, MI=False, pred12v1=False):
        super().__init__()
        self.grl = GradientReversalLayer()

        if AS:
            self.network = nn.Sequential(
                nn.Linear(latent_dim-pred2, 100),
                nn.LeakyReLU(0.01),
                nn.Linear(100, classes), 
            )
        elif MI:
            self.network = nn.Sequential(
                nn.Linear(latent_dim-pred2, 100), 
                nn.LeakyReLU(0.01),
                nn.Linear(100, classes), 
                nn.Softmax(dim=1))
        elif three_class:
            self.network = nn.Sequential(
                nn.Linear(latent_dim-4, 100),
                nn.LeakyReLU(0.01),
                nn.Linear(100, classes), 
                nn.Softmax(dim=1))
        elif lead1:
            self.network = nn.Sequential(
                nn.Linear(latent_dim-pred2, 100),
                nn.LeakyReLU(0.01),
                nn.Linear(100, classes))
        elif pred12v1:
            self.network = nn.Sequential(
                nn.Linear(latent_dim, 100), 
                nn.LeakyReLU(0.01),
                LinearResidualBlock(100),
                nn.Linear(100, classes))
        else:
            self.network = nn.Sequential(
                nn.Linear(latent_dim-pred2, 3), 
                nn.LeakyReLU(0.01),
                nn.Linear(3, classes),
                nn.Flatten())
        
    def forward(self, x):
        x = self.grl(x)
        x = self.network(x)
        return x

    

class VAE_linear12(nn.Module):
    """Fully-connected VAE for 12-lead ECG.

    The first `pred` dims of z carry the regression target (LVF/MI); the
    remaining dims carry the non-label signal. The decoder expands the
    latent back out to 12 x 400.
    """
    def __init__(self, latent_dim=64, pred=1):
        super().__init__()
        
        self.pred = pred
        
        self.encoder = nn.Sequential(
            nn.Linear(400, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            LinearResidualBlock(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(600, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            LinearResidualBlock(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.01),
        )
        
        self.z_mean = nn.Linear(50, latent_dim)
        self.z_log_var = nn.Linear(50, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            LinearResidualBlock(100),
            nn.Linear(100, 600),
            nn.LeakyReLU(0.01),
            Reshape(-1, 12, 50),
            nn.Linear(50, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            LinearResidualBlock(100),
            nn.Linear(100, 400),
        )
        
        self.regressor = nn.Linear(self.pred, 4)
        # self.final = nn.Softmax(dim=1)
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(*z_mu.size()).to(z_mu.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z
        
    def forward(self, x):
        x = x.view(-1, 12, 400)
        encoded = self.encoder(x)

        z_mean, z_log_var = self.z_mean(encoded), self.z_log_var(encoded)
        z_log_var = torch.clamp(z_log_var, min=-5, max=5)
        
        z = self.reparameterize(z_mean, z_log_var)
        
        pred_train = self.regressor(z[:, :self.pred])
        pred_val = self.regressor(z_mean[:, :self.pred])
        
        decoded = self.decoder(z)        

        return decoded, z_mean, z_log_var, pred_train, pred_val
        
    def decode(self, x):
        decoded = self.decoder(x)
        return decoded      


class VAE_conv12(nn.Module):
    """Convolutional VAE for 12-lead ECG.

    Input treated as a 2D image (1, 12, 400) where the 12 leads form one
    spatial axis; uses 2D residual blocks. First `pred` latent dims are
    the label-relevant component.
    """
    def __init__(self, latent_dim=64, pred=1, output_pred=4):
        super().__init__()
        
        downsample=True
        self.pred = pred
        stride = (1, 2)
        self.sig_dim = 13
        self.final_dim = 385
        input_dim = 400
        self.flatten_dim = 12*64*self.sig_dim
        k_size = 3
        padding0 = 1
        
        self.encoder = nn.Sequential(
            ResidualBlock(1, 8, stride=stride, kernel_size=(1, 5), padding=(0, 2), downsample=downsample),
            ResidualBlock(8, 16, stride=stride, kernel_size=(1, 5), padding=(0, 2), downsample=downsample),
            ResidualBlock(16, 16, stride=(1, 1), kernel_size=(1, 5), padding=(0, 2)),
            ResidualBlock(16, 32, stride=stride, kernel_size=(1, 5), padding=(0, 2), downsample=downsample),
            ResidualBlock(32, 32, stride=(1, 1), kernel_size=(k_size, 5), padding=(padding0, 2)),
            ResidualBlock(32, 64, stride=stride, kernel_size=(k_size, 5), padding=(padding0, 2), downsample=downsample),
            ResidualBlock(64, 64, stride=stride, kernel_size=(k_size, 5), padding=(padding0, 2), downsample=downsample),
        )

        self.z_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.z_log_var = nn.Linear(self.flatten_dim, latent_dim)
        self.flat = nn.Flatten()
        
        self.decoder_in = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            ResidualBlockTranspose(64, 64, stride=stride, kernel_size=(k_size, 5), padding=(padding0, 2), upsample=downsample),
            ResidualBlockTranspose(64, 32, stride=stride, kernel_size=(k_size, 5), padding=(padding0, 2), upsample=downsample),
            ResidualBlockTranspose(32, 32, stride=(1, 1), kernel_size=(k_size, 5), padding=(padding0, 2)),  
            ResidualBlockTranspose(32, 16, stride=stride, kernel_size=(1, 5), padding=(0, 2), upsample=downsample),
            ResidualBlockTranspose(16, 16, stride=(1, 1), kernel_size=(1, 5), padding=(0, 2)),
            ResidualBlockTranspose(16, 8, stride=stride, kernel_size=(1, 5), padding=(0, 2), upsample=downsample),
            ResidualBlockTranspose(8, 1, stride=stride, kernel_size=(1, 5), padding=(0, 2), upsample=downsample),
            nn.Linear(self.final_dim, input_dim),
        )
        
        self.regressor = nn.Linear(self.pred, output_pred)
        # self.final = nn.Softmax(dim=1)
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(*z_mu.size()).to(z_mu.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z
        
    def forward(self, x):
        x = x.view(-1, 1, 12, 400)
        encoded = self.encoder(x)
        
        encoded = self.flat(encoded)

        z_mean, z_log_var = self.z_mean(encoded), self.z_log_var(encoded)
        z_log_var = torch.clamp(z_log_var, min=-5, max=5)
        
        z = self.reparameterize(z_mean, z_log_var)
        
        pred_train = self.regressor(z[:, :self.pred])
        pred_val = self.regressor(z_mean[:, :self.pred])
        
        decoded = self.decoder_in(z)
        decoded = decoded.view(-1, 64, 12, self.sig_dim)
        
        decoded = self.decoder(decoded)        

        return decoded, z_mean, z_log_var, pred_train, pred_val
        
    def decode(self, x):
        decoded = self.decoder_in(x)
        decoded = decoded.view(-1, 64, 12, self.sig_dim)
        decoded = self.decoder(decoded)
        return decoded      


class VAE_linear1to12(nn.Module):
    """Cross-lead VAE: 1-lead -> 12-lead and 12-lead -> 1-lead generation
    using two parallel encoders and two decoders (lead-matching).
    """
    def __init__(self, leads=12, latent_dim=64):
        super().__init__()
        
        self.leads = leads
        self.flatten_dim = 64*4*leads

        self.encoder1 = nn.Sequential(
            nn.Linear(400, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.01),
        )
        
        self.encoder12 = nn.Sequential(
            nn.Linear(400, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.01),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 100), 
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            nn.Linear(100, 1))
        
        self.z_mean1 = nn.Linear(50, latent_dim)
        self.z_log_var1 = nn.Linear(50, latent_dim)
        
        self.z_mean12 = nn.Linear(50, latent_dim)
        self.z_log_var12 = nn.Linear(50, latent_dim)
        
        self.decoderto12 = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            nn.Linear(100, 400),

        )
        
        self.decoderto1 = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50, 100),
            nn.LeakyReLU(0.01),
            LinearResidualBlock(100),
            nn.Linear(100, 400),
   
        )
        
    def gradient_reverse(self, x, lambda_):
        return GradientReversal.apply(x, lambda_)
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(*z_mu.size()).to(z_mu.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z
        
    def forward(self, x):
        x = x.view(-1, 400)
        encoded1 = self.encoder1(x)
        encoded12 = self.encoder12(x)

        z_mean1, z_log_var1 = self.z_mean1(encoded1), self.z_log_var1(encoded1)
        z_log_var1 = torch.clamp(z_log_var1, min=-5, max=5)
        
        z_mean12, z_log_var12 = self.z_mean12(encoded12), self.z_log_var12(encoded12)
        z_log_var12 = torch.clamp(z_log_var12, min=-5, max=5)        
        
        z1 = self.reparameterize(z_mean1, z_log_var1)
        z12 = self.reparameterize(z_mean12, z_log_var12)
        
        decoded1to12 = self.decoderto12(z1)
        decoded12to12 = self.decoderto12(z12)
        decoded1to1 = self.decoderto1(z1)

        return decoded1to12, decoded12to12, decoded1to1, z_mean1, z_log_var1, z_mean12, z_log_var12
    
    def decode(self, x):

        decoded1to12 = self.decoderto12(x)
        decoded12tot12 = self.decoderto12(x)
        decoded1to1 = self.decoderto1(x)
        return decoded1to12, decoded12tot12, decoded1to1      
