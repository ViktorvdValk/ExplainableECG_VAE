"""Main entry point for the ECG VAE training pipeline.

Builds the data loaders and runs Trainer_12 over 5 cross-validation
folds (12-lead or 1-lead ECGs, LVF/MI labels).

Run with:  python main_12.py
"""

import torch
import typer
from utils.helper_train2 import Trainer_12
from utils.helper_load import generate_12_loader, generate_1_loader
from utils.helper_utils import git_autocommit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def run_test(version='v1', batch_size=100, num_epochs=50, cv=0, components=14, beta_ls=0.001, beta_nls=0.001, gamma=2, pred=1, patience=15, lr=0.001, save=False, 
             labels=['ncvd', 'pci_cabg'], leads12=12, binary=False, prev=False, model_str='', first_lead_only=False, all=False):
    """Configure and run a single training experiment.

    Builds the matching data loader (12-lead vs 1-lead), instantiates
    the Trainer_12 with the given hyper-parameters, trains the model
    and optionally saves the weights to disk.

    Args:
        version: version tag used for logging and model saving
        batch_size: mini-batch size for training
        num_epochs: maximum number of training epochs
        cv: cross-validation fold index (also random seed)
        components: dimensionality of the VAE latent space
        beta_ls: weight of the label-related KL divergence term
        beta_nls: weight of the non-label KL divergence term
        gamma: weight of the prediction loss
        pred: number of "label-predicting" latent dimensions
        patience: early-stopping patience (epochs w/o improvement)
        lr: learning rate for the optimizer
        save: whether to save the model weights after training
        labels: target label column(s) to predict
        leads12: 12 for the 12-lead setup, else 1-lead
        binary: binary target (True) vs 4-class ordinal target (False)
        prev: resume from a previously saved model (weights in model_str)
        model_str: path of previous weights to load (when prev=True)
        first_lead_only: use only the first lead (repeated 12x)
        all: use the full dataset (True) or the saved train split (False)
    """

    
    print('Getting loader ready..')
    # if leads12 == 12:
    #     generate_12_loader(cv=cv, batch_size=batch_size, labels=labels, binary=binary, first_lead_only=first_lead_only, all=all)
    # else:
    #     generate_1_loader(cv=cv, batch_size=batch_size, labels=labels, binary=binary)
    if binary:
        out_pred = 1
    else:
        out_pred = 4
    print('Start trainer')
    trainer = Trainer_12(version=version, batch_size=batch_size, learning_rate=lr, pred=pred, out_pred=out_pred, pca_init=False, cv=cv, 
                         beta_ls=beta_ls, beta_nls=beta_nls, gamma=gamma, patience=patience, components=components,
                         model_shape='conv', labels=labels, leads12=leads12, binary=binary, prev=prev, model_str=model_str,
                         first_lead_only=first_lead_only, all=all)
                                    
    trainer.train_12(num_epochs=num_epochs)

    if save:
        trainer.save_model()
        
def main(save: bool = True, version: str = 'v_last', batch_size: int = 300, num_epochs: int = 500):
    """Entry point (typer CLI): run the 5-fold training loop.

    Auto-commits the current code state to git, then trains one model
per fold (cv = 0..4) on the 'lvf' label with a 1-lead setup.
    Exposed via typer so hyper-parameters can be overridden on the CLI.
    """

    git_autocommit(version=version)
    for cv in [0,1,2,3,4]:
        run_test(version=version, batch_size=batch_size, num_epochs=num_epochs, cv=cv, components=600, beta_ls=0.00, beta_nls=0.0001, gamma=3, pred=1, patience=15, lr=0.0001, 
                        save=True, labels=['lvf'], leads12=1, binary=False, prev=True, first_lead_only=False, all=False)
           
if __name__ == "__main__":
    typer.run(main)

