"""Loss functions and learning-rate schedules used by the ECG VAE trainers.

    - ResettableReduceLROnPlateau: ReduceLROnPlateau that can be reset to the
      initial LR (used after fine-tuning restarts)
    - ContrastiveLoss_ECG: margin-based contrastive loss over multiple label dims
    - SA_RPS_Loss: soft all-rank-pair-sorting loss for ordinal classes
    - ordinal_to_cumsum / one_hot_encoding: label encoders for classification
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


RECON_DIST = ["bernoulli", "laplace", "gaussian"]

class ResettableReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau with a reset() to restore the initial LR + counters."""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(ResettableReduceLROnPlateau, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                                                          threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)

        self.optimizer = optimizer

    def reset(self, initial_lr):
        """Reset best-loss and counters, and restore the initial learning rate."""

        self.best = float('inf') if self.mode == 'min' else -float('inf')
        self.num_bad_epochs = 0
        self.last_epoch = -1
        self._reset2(initial_lr)

    def _reset2(self, initial_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

class ContrastiveLoss_ECG(nn.Module):
    """Contrastive loss: pull same-label outputs together, push different apart."""

    def __init__(self, margin=1.0):
        super(ContrastiveLoss_ECG, self).__init__()
        self.margin = margin

    def forward(self, outputs1, outputs2, labels1, labels2):
        # Calculate Euclidean distance between the output pairs (shape: [batch_size])
        
        euclidean_distance = F.pairwise_distance(outputs1, outputs2)
        # Normalise by the output dimension so the loss magnitude is
        # comparable across embedding sizes.
        
        # make euclidean_distance dependent on the output dim
        euclidean_distance = torch.pow(torch.pow(euclidean_distance, 2) / outputs1.shape[1], .5)
        
        # labels is of shape [batch_size, 3] while euclidean_distance is [batch_size]
        # We need to compute the loss for each of the three label dimensions
        loss = 0.0

        for i in [0,2]:
            weights = labels1[:, i] == labels2[:, i]  # Select the label for the i-th dimension

            # Use the label as the weight directly
            weights = weights.float()

            # Calculate positive loss for label > 0
            positive_loss = weights * torch.pow(euclidean_distance, 2)

            # Calculate negative loss for label = 0
            negative_loss = (weights == 0).float() * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

            # Add this loss component to the total loss
            loss += torch.mean(positive_loss + negative_loss)
        
        # Return the average loss across all labels
        return loss


def ordinal_to_cumsum(labels, num_classes):
    binary_labels = np.zeros((len(labels), num_classes-1))
    for i, label in enumerate(labels):
        for j in range(num_classes-1):
            binary_labels[i, j] = 0 if label > j else 1
    return torch.tensor(binary_labels, dtype=torch.float32)


def one_hot_encoding(labels, num_classes):
    binary_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):       
        binary_labels[i, int(label.cpu().numpy())] = 1
    return torch.tensor(binary_labels, dtype=torch.float32)


class SA_RPS_Loss(nn.Module):
    """Soft-APRS (all rank-pair sorting) loss for ordinal / multi-class targets."""

    def __init__(self, classes=4, combined=False):
        super(SA_RPS_Loss, self).__init__()
        self.classes = classes
        self.combined = combined

    def forward(self, predictions, targets):
        predictions_sm = F.softmax(predictions, dim=1)
        y = ordinal_to_cumsum(targets, self.classes).to(predictions.device)
        p = torch.cumsum(predictions_sm, dim=1).to(predictions.device)
        k = torch.tensor(y.shape[1] - 1, device=p.device)
        saRPS = torch.empty(y.shape[0], dtype=torch.float32, device=p.device)
        for i, (y_i, p_i) in enumerate(zip(y, p)):
            saRPS[i] = (torch.sum(torch.abs(p_i[:-1] - y_i)) / k) ** 2
        
        if self.combined:
            loss_f = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_f(predictions, targets.squeeze().long())
            return loss.mean() + saRPS.mean()
        else:
            return saRPS.mean()

