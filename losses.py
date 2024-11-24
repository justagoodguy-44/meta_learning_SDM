import torch
import numpy as np


def get_species_weights(y_train, species_weights_method="inversely_proportional"):
    if species_weights_method == "inversely_proportional":
        species_weights = y_train.shape[0] / (y_train.sum(0) + 1e-5)
    elif species_weights_method == "inversely_proportional_clipped":
        species_weights = y_train.shape[0] / (y_train.sum(0) + 1e-5)
        species_weights = np.clip(species_weights, 0.25, 4)
    elif species_weights_method == "inversely_proportional_sqrt":
        species_weights = np.sqrt(y_train.shape[0] / (y_train.sum(0) + 1e-5))
    elif species_weights_method == "class_balanced":
        beta = 0.999
        species_weights = (1 - beta)/ (1 - beta ** (y_train.sum(0) + 1e-5))
    elif species_weights_method == "uniform":
        species_weights = 2 * np.ones(y_train.shape[1])
    else:
        raise ValueError("species_weights_method must be 'inversely_proportional', 'inversely_proportional_clipped', 'inversely_proportional_sqrt', 'class_balanced' or 'uniform'")
    return species_weights
        

def full_weighted_loss(pred_x, y, pred_bg, species_weights=None):
    
    batch_size = pred_x.size(0)
        
    # loss at data location
    if species_weights is not None:
        loss_dl_pos = (log_loss(pred_x) * y * species_weights.repeat((batch_size, 1))).mean()
        loss_dl_neg = (log_loss(1 - pred_x) * (1 - y) * (species_weights/(species_weights - 1)).repeat((batch_size, 1))).mean() # or 1/(1 - 1/species_weights)
    else:
        loss_dl_pos = (log_loss(pred_x) * y).mean()
        loss_dl_neg = (log_loss(1 - pred_x) * (1 - y)).mean() # or 1/(1 - 1/species_weights)
    
    # loss at random location
    loss_rl = log_loss(1 - pred_bg).mean()
        
    return loss_dl_pos, loss_dl_neg, loss_rl


def log_loss(pred):
    return -torch.log(pred + 1e-5)


def total_loss(loss_dl_pos, loss_dl_neg, loss_rl, lambda1, lambda2, lambda3):
    return lambda1 * loss_dl_pos + lambda2 * loss_dl_neg + lambda3 * loss_rl