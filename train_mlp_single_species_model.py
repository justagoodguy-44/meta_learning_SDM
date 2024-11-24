
import argparse

import numpy as np
import torch
import wandb
import verde as vd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load

from training_helpers import *
from data_helpers import get_data_single_species, get_species_list, split_test_data_for_cross_validation, split_train_data_for_cross_validation
from losses import full_weighted_loss, get_species_weights
from utils import *
from log_helpers import *


def main():

    run_type = RunType.SINGLE_SPECIES_MLP
    
    arg_parser = get_mlp_arg_parser()
    args = arg_parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"current device: {device}")

    # Region of interest
    region = args.region

    hparams = read_params_from_json(args.hparams_path)
    
    # We fix the splits
    seed_everything(hparams["seed_split"])

    # To store the results
    df_results = pd.DataFrame()

    species_in_region = get_species_list(region, remove=False)
    for i, species in enumerate(species_in_region):

        x_train, y_train, coordinates_train, x_test, y_test, coordinates_test, bg = get_data_single_species(region, species, co_occurrence=hparams["co_occurrence"], valavi=hparams["valavi"])

        # Scale data
        x_train, x_test, bg = scale_data(x_train, x_test, bg, args.scaler_path)

        # Cross validation
        if hparams["cross_validation_using_test"]:
            # Use a subset of the test data as validation
            x_test, y_test, x_val, y_val = split_test_data_for_cross_validation(x_test, y_test, coordinates_test, hparams)
            bg_train = bg
            bg_val = None
        else:
            # Use the classic subset of training data as validation
            x_train, y_train, x_val, y_val, bg_train, bg_val = split_train_data_for_cross_validation(x_train, y_train, coordinates_train, bg, hparams)
        
        # If we want to store results on validation and test set during training
        training_val_and_test_results_path = args.training_val_and_test_results_path
        if training_val_and_test_results_path is not None:
            training_results_val = []
            training_results_test = []
        else:
            training_results_val = None
            training_results_test = None

        num_covariates = x_train.shape[1]
        lambda_1 = 1
        lambda_2 = 0.8
        lambda_3 = 1 - lambda_2
        num_species = 1

        # Initialize the config as the hyperparameters and add run-specific values
        config = hparams.copy()
        config["region"] = region
        config["num_covariates"] = num_covariates
        config["device"] = device
        config["lambda_1"] = lambda_1
        config["lambda_2"] = lambda_2
        config["lambda_3"] = lambda_3
        config["loss_fn"] = full_weighted_loss
        config["species"] = species
        config["species_idx"] = i
        config["num_species"] = num_species
        config["run_type"] = run_type

        data_holder = DataHolder(x_train, y_train, bg_train, x_test, y_test, x_val, y_val, bg_val)
        model = train_MLP(data_holder, config, auc_roc_val_all=training_results_val, auc_roc_test_all=training_results_test)

        # evaluate and log
        y_pred = predict(model, x_test, device)
        aucs = evaluate_results(y_test, y_pred)
        df_aucs = pd.DataFrame(aucs, index=[species])
        df_results = pd.concat([df_results, df_aucs])

        # Save the training results
        if training_val_and_test_results_path is not None:
            log_training_results(run_type, args.training_val_and_test_results_path, training_results_val, region, species, "val")
            log_training_results(run_type, args.training_val_and_test_results_path, training_results_test, region, species, "test")

        # Save the model
        if args.save_model_path is not None:
            save_model(model, args.save_model_path, run_type, region)

    log_run(run_type, config, args.log_path, df_results)



if __name__ == '__main__':
    main()