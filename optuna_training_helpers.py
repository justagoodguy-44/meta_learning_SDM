import os
import random
import numpy as np
import torch
import wandb
import learn2learn as l2l
import optuna
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from tqdm import trange

from data_helpers import SpeciesDataset, get_data_one_region, get_region_list
from data_holder import DataHolder
from optuna_data_holder import OptunaDataHolder
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from models import MLP
from losses import full_weighted_loss, get_species_weights, total_loss
from prg_corrected import create_prg_curve, calc_auprg
from utils import indices_to_one_hot, one_hot_to_indices, scale_data
from training_helpers import predict, evaluate_results, train_maml


def optuna_trials_multi_species(data_holder_all_regions, configs, optuna_data_holder:OptunaDataHolder, trial, base_models=None):
    
    lr_low, lr_high = optuna_data_holder.get_interval("lr")
    lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
    dropout_low, dropout_high = optuna_data_holder.get_interval("dropout")
    dropout = trial.suggest_float("dropout", dropout_low, dropout_high, log=True)
    weight_decay_low, weight_decay_high = optuna_data_holder.get_interval("weight_decay")
    weight_decay = trial.suggest_float("weight_decay", weight_decay_low, weight_decay_high, log=True)
    num_layers_low, num_layers_high = optuna_data_holder.get_interval("num_layers")
    num_layers = trial.suggest_int("num_layers", num_layers_low, num_layers_high)
    layer_width_low, layer_width_high = optuna_data_holder.get_interval("layer_width")
    layer_width = trial.suggest_int("layer_width", layer_width_low, layer_width_high)


    
    regions = get_region_list()
    species_weights_all_regions = []
    trainloaders = []
    models = []
    optimizers = []
    schedulers = []
    for i in range(len(regions)):
        config = configs[i]

        data_holder = data_holder_all_regions[i]
        
        x_train = data_holder.get("x_train")
        y_train = data_holder.get("y_train")
        bg_train = data_holder.get("bg_train")

        lr_lambda = lambda epoch: config["learning_rate_decay"] ** epoch
        
        species_weights_all_regions.append(torch.tensor(get_species_weights(y_train), dtype=torch.float32).to(config["device"]))
        
        trainset = SpeciesDataset(x_train, y_train, bg_train)
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
        trainloaders.append(trainloader)
        
        if base_models == None:
            model = MLP(input_size=x_train.shape[1], output_size=config["num_species"], num_layers=num_layers, 
                        width=layer_width, dropout=dropout)
        else:
            model = base_models[i]
        model.train()
        model.to(config["device"])
        models.append(model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizers.append(optimizer)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        schedulers.append(scheduler)

   
    loss_fn = config["loss_fn"]
    auc_rocs = []
    best_mean_auc_roc = 0
    patience = 3
    impatience = 0
    still_patient = True
    epoch_counter = 0
    for i in trange(config["epochs"]):
        if not still_patient:
            break
        auc_rocs.clear()
        epoch_counter += 1
        for j in range(len(regions)):
            if not still_patient:
                break
            model = models[j]
            species_weights = species_weights_all_regions[j]
            trainloader = trainloaders[j]
            optimizer = optimizers[j]
            scheduler = schedulers[j]
            data_holder = data_holder_all_regions[j]
            config = configs[j]
            x_val = data_holder.get("x_val").to("cpu")
            y_val = data_holder.get("y_val").to("cpu")

            if i > 0: 
                scheduler.step()

            for x_batch, y_batch, bg_batch in trainloader:
                # Forward pass
                output = torch.sigmoid(model(torch.cat((x_batch, bg_batch), 0)))
                pred_x = output[:len(x_batch)]
                pred_bg = output[len(x_batch):]
                loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_x, y_batch, pred_bg, species_weights)
                # Compute total loss
                train_loss = total_loss(loss_dl_pos, loss_dl_neg, loss_rl, config["lambda_1"], config["lambda_2"], config["lambda_3"])
                # Backward pass
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                model.eval()
                aucs = evaluate_results(model, x_val, y_val, config)
                model.train()
                auc_roc = aucs["auc_roc"]
                auc_rocs.append(auc_roc)

        mean_auc_roc = np.nanmean(auc_rocs)
        trial.report(mean_auc_roc, i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        #early stopping
        if mean_auc_roc > best_mean_auc_roc:
            best_mean_auc_roc = mean_auc_roc
            impatience = 0
        else:
            impatience += 1
            if impatience >= patience:
                still_patient = False
    
    print(f"Went through {epoch_counter} out of {config['epochs']} epochs")
    return np.nanmean(auc_rocs)




def optuna_trials_single_species(data_holder_all_regions, configs, optuna_data_holder:OptunaDataHolder, trial, base_models=None):
    
    lr_low, lr_high = optuna_data_holder.get_interval("lr")
    lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
    dropout_low, dropout_high = optuna_data_holder.get_interval("dropout")
    dropout = trial.suggest_float("dropout", dropout_low, dropout_high, log=True)
    weight_decay_low, weight_decay_high = optuna_data_holder.get_interval("weight_decay")
    weight_decay = trial.suggest_float("weight_decay", weight_decay_low, weight_decay_high, log=True)
    num_layers_low, num_layers_high = optuna_data_holder.get_interval("num_layers")
    num_layers = trial.suggest_int("num_layers", num_layers_low, num_layers_high)
    layer_width_low, layer_width_high = optuna_data_holder.get_interval("layer_width")
    layer_width = trial.suggest_int("layer_width", layer_width_low, layer_width_high)
    optuna_params = {
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "num_layers": num_layers,
        "layer_width": layer_width
    }

    return _trial_single_species(data_holder_all_regions, configs, base_models, optuna_params)


def _trial_single_species(data_holder_all_regions, configs, base_models, optuna_params:dict):
    lr = optuna_params["lr"]
    dropout = optuna_params["dropout"]
    weight_decay = optuna_params["weight_decay"]
    num_layers = optuna_params["num_layers"]
    layer_width = optuna_params["layer_width"]

    #number of species that will be used to train and evaluate different models
    num_species_per_region = len(data_holder_all_regions[0])
    
    regions = get_region_list()
    species_weights_all_regions = []
    trainloaders = []
    models = []
    optimizers = []
    schedulers = []

    for i in range(len(regions)):

        species_weights_region = []
        trainloaders_region = []
        models_region = []
        optimizers_region = []
        schedulers_region = []

        config = configs[i][0]

        for j in range(num_species_per_region):
            data_holder = data_holder_all_regions[i][j]
            
            x_train = data_holder.get("x_train")
            y_train = data_holder.get("y_train")
            bg_train = data_holder.get("bg_train")

            lr_lambda = lambda epoch: config["learning_rate_decay"] ** epoch

            species_weights_region.append(torch.tensor(get_species_weights(y_train), dtype=torch.float32).to(config["device"]))
                        
            trainset = SpeciesDataset(x_train, y_train, bg_train)
            trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
            trainloaders_region.append(trainloader)
            
            if base_models==None:
                model = MLP(input_size=x_train.shape[1], output_size=config["num_species"], num_layers=num_layers, 
                            width=layer_width, dropout=dropout)
            else:
                model = base_models[i]
                model.change_dropout(dropout)

            model.train()
            model.to(config["device"])
            models_region.append(model)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            optimizers_region.append(optimizer)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            schedulers_region.append(scheduler)

        species_weights_all_regions.append(species_weights_region)
        trainloaders.append(trainloaders_region)
        models.append(models_region)
        optimizers.append(optimizers_region)
        schedulers.append(schedulers_region)
   
    loss_fn = config["loss_fn"]
    auc_rocs = []
    best_mean_auc_roc = 0
    patience = 3
    impatience = 0
    still_patient = True
    epoch_counter = 0
    for i in trange(config["epochs"]):
        if not still_patient:
            break
        epoch_counter += 1
        auc_rocs.clear()
        for j in range(len(regions)):
            if not still_patient:
                break
            for k in range(num_species_per_region):
                if not still_patient:
                    break
                model = models[j][k]
                species_weights = species_weights_all_regions[j][k]
                trainloader = trainloaders[j][k]
                optimizer = optimizers[j][k]
                scheduler = schedulers[j][k]
                data_holder = data_holder_all_regions[j][k]
                config = configs[j][k]
                x_val = data_holder.get("x_val").to("cpu")
                y_val = data_holder.get("y_val").to("cpu")

                if i > 0: 
                    scheduler.step()

                for x_batch, y_batch, bg_batch in trainloader:
                    # Forward pass
                    output = torch.sigmoid(model(torch.cat((x_batch, bg_batch), 0)))
                    pred_x = output[:len(x_batch)]
                    pred_bg = output[len(x_batch):]
                    loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_x, y_batch, pred_bg, species_weights)
                    # Compute total loss
                    train_loss = total_loss(loss_dl_pos, loss_dl_neg, loss_rl, config["lambda_1"], config["lambda_2"], config["lambda_3"])
                    # Backward pass
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    model.eval()
                    y_pred = predict(model, x_val, config["device"])
                    aucs = evaluate_results(y_val, y_pred, config["species_idx"])
                    model.train()
                    auc_roc = aucs["auc_roc"]
                    auc_rocs.append(auc_roc)

        mean_auc_roc = np.nanmean(auc_rocs)
        
        #early stopping
        if mean_auc_roc > best_mean_auc_roc:
            best_mean_auc_roc = mean_auc_roc
            impatience = 0
        else:
            impatience += 1
            if impatience >= patience:
                still_patient = False
    
    print(f"Went through {epoch_counter} out of {config['epochs']}")
    return np.nanmean(auc_rocs)




def optuna_trials_maml(data_holder_all_regions_meta_train, data_holder_all_regions_meta_test, configs, optuna_data_holder:OptunaDataHolder, trial):
    
    lr_low, lr_high = optuna_data_holder.get_interval("lr")
    lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
    dropout_low, dropout_high = optuna_data_holder.get_interval("dropout")
    dropout = trial.suggest_float("dropout", dropout_low, dropout_high, log=True)
    weight_decay_low, weight_decay_high = optuna_data_holder.get_interval("weight_decay")
    weight_decay = trial.suggest_float("weight_decay", weight_decay_low, weight_decay_high, log=True)
    num_layers_low, num_layers_high = optuna_data_holder.get_interval("num_layers")
    num_layers = trial.suggest_int("num_layers", num_layers_low, num_layers_high)
    layer_width_low, layer_width_high = optuna_data_holder.get_interval("layer_width")
    layer_width = trial.suggest_int("layer_width", layer_width_low, layer_width_high)
    outer_loop_iters_low, outer_loop_iters_high = optuna_data_holder.get_interval("outer_loop_iters")
    outer_loop_iters = trial.suggest_int("outer_loop_iters", outer_loop_iters_low, outer_loop_iters_high)
    fast_lr_low, fast_lr_high = optuna_data_holder.get_interval("fast_lr")
    fast_lr = trial.suggest_float("fast_lr", fast_lr_low, fast_lr_high, log=True)
    meta_lr_low, meta_lr_high = optuna_data_holder.get_interval("meta_lr")
    meta_lr = trial.suggest_float("meta_lr", meta_lr_low, meta_lr_high, log=True)
    

    optuna_params = {
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "num_layers": num_layers,
        "layer_width": layer_width,
    }
  
    # Meta training

    regions = get_region_list()
    meta_train_models = []

    for i in range(len(regions)):

        config_meta_train = configs[i][0]
        config_meta_train["lr"] = lr
        config_meta_train["dropout"] = dropout
        config_meta_train["weight_decay"] = weight_decay
        config_meta_train["num_layers"] = num_layers
        config_meta_train["layer_width"] = layer_width
        config_meta_train["outer_loop_iters"] = outer_loop_iters
        config_meta_train["fast_lr"] = fast_lr
        config_meta_train["meta_lr"] = meta_lr
    
        meta_train_models.append(train_maml(data_holder_all_regions_meta_train[i], config_meta_train))

    # Meta testing
    return _trial_single_species(data_holder_all_regions_meta_test, configs, meta_train_models, optuna_params)
