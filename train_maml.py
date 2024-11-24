
import numpy as np
import torch
import wandb

from training_helpers import *
from data_helpers import *
from losses import full_weighted_loss, get_species_weights
from utils import *
from log_helpers import *
from data_holder import DataHolder

def main():

    run_type = RunType.MAML

    arg_parser = get_meta_learning_arg_parser()
    args = arg_parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"current device: {device}")

    # Region of interest
    region = args.region

    hparams = read_params_from_json(args.hparams_path)
    
    # We fix the splits
    seed_everything(hparams["seed_split"])

    # Get the data
    x_train, y_train, coordinates_train, x_test, y_test, coordinates_test, bg = get_data_one_region(region, co_occurrence=False, valavi=hparams["valavi"])

    # Scale the data
    x_train, x_test, bg = scale_data(x_train, x_test, bg, args.scaler_path)

        
    all_species_in_region_count = y_train.shape[1]

    # If interested only in the fine-tuning of a specific species
    specific_species = args.specific_species
    if specific_species is not None:
        specific_species_idx = get_species_idx_from_name(specific_species, region)
        num_samples = args.num_samples
        specific_species_samples_idx_in_train = np.array(get_species_samples_idx(y_train, specific_species_idx))
        #Keep only desired number of samples of this species
        specific_species_samples_idx_to_keep = sorted(random.sample(range(len(specific_species_samples_idx_in_train)), num_samples))
        specific_species_samples_idx_in_train_to_keep = specific_species_samples_idx_in_train[specific_species_samples_idx_to_keep]
        sample_idx_to_discard = np.array([sample_idx for sample_idx in specific_species_samples_idx_in_train if sample_idx not in specific_species_samples_idx_in_train_to_keep])
        all_samples_idx_to_keep = np.setdiff1d(np.arange(y_train.shape[0]), sample_idx_to_discard)
        print(f"num samples before reduction: {len(x_train)}")
        x_train = x_train[all_samples_idx_to_keep]
        y_train = y_train[all_samples_idx_to_keep]
        coordinates_train = coordinates_train[all_samples_idx_to_keep]
        print(f"num samples after reduction: {len(x_train)}")

     # Cross validation
    if hparams["cross_validation_using_test"]:
        # Use a subset of the test data as validation
        x_test, y_test, x_val, y_val = split_test_data_for_cross_validation(x_test, y_test, coordinates_test, hparams)
        bg_train = bg
        bg_val = None
    else:
        # Use the classic subset of training data as validation
        x_train, y_train, x_val, y_val, bg_train, bg_val = split_train_data_for_cross_validation(x_train, y_train, coordinates_train, bg, hparams)

    # Keep only species with minimum number of samples (multiply by 2 the number of shots for query set)
    min_samples = 2*hparams["shots"]
    enough_samples_row_idx, enough_samples_column_idx = keep_only_species_with_enough_samples(y_train, min_samples)
    ratio_data_kept = len(enough_samples_row_idx) / len(y_train)
    x_train = x_train[enough_samples_row_idx] 
    y_train = y_train[enough_samples_row_idx]
    # Have to also remove columns of other species for y
    y_train = y_train[:,enough_samples_column_idx]
    coordinates_train = coordinates_train[enough_samples_row_idx]
    species_kept_count = len(np.where(np.sum(y_train, axis=0) > 0)[0])
    print(f"In region {region}, able to train on {species_kept_count} out of {all_species_in_region_count} species, and {ratio_data_kept * 100}% of data, which meet the requirement of having more than {min_samples} samples")

    num_covariates = x_train.shape[1]
    lambda_1 = 1
    lambda_2 = 0.8
    lambda_3 = 1 - lambda_2
    num_species = y_train.shape[1]

    # Initialize the config as the hyperparameters and add run-specific values
    config = hparams.copy()
    config["region"] = region
    config["num_covariates"] = num_covariates
    config["device"] = device
    config["lambda_1"] = lambda_1
    config["lambda_2"] = lambda_2
    config["lambda_3"] = lambda_3
    config["loss_fn"] = full_weighted_loss
    config["num_species"] = num_species
    config["run_type"] = run_type

    # ========= META TRAINING ==========
    if args.load_model_path == None:
        data_holder = DataHolder(x_train, y_train, bg_train, x_test, y_test, x_val, y_val, bg_val)
        base_model = train_maml(data_holder, config)
        # Save the model
        if args.save_model_path is not None:
            save_model(base_model, args.save_model_path, RunType.MAML, region) 
    
    else:
        base_model = torch.load(args.load_model_path)


    # ========= META TESTING ==========

    # To store the results
    df_results = pd.DataFrame()

    species_in_region = get_species_list(region, remove=False)
    for i, species in enumerate(species_in_region):
        if specific_species is None or species==specific_species:

            config["species"] = species
            
            x_train, y_train, coordinates_train, x_test, y_test, coordinates_test, bg = get_data_single_species(region, species, co_occurrence=hparams["co_occurrence"], valavi=hparams["valavi"])
            
            # Scale data
            x_train, x_test, bg = scale_data(x_train, x_test, bg, args.scaler_path)

            # Once again only keep the desired number of samples of the species of interest
            if species==specific_species:
                specific_species_samples_idx_in_train = np.array(get_species_samples_idx(y_train))
                #Keep only desired number of samples of this species
                specific_species_samples_idx_in_train_to_keep = specific_species_samples_idx_in_train[specific_species_samples_idx_to_keep]
                sample_idx_to_discard = np.array([sample_idx for sample_idx in specific_species_samples_idx_in_train if sample_idx not in specific_species_samples_idx_in_train_to_keep])
                all_samples_idx_to_keep = np.setdiff1d(np.arange(y_train.shape[0]), sample_idx_to_discard)
                x_train = x_train[all_samples_idx_to_keep]
                y_train = y_train[all_samples_idx_to_keep]
                coordinates_train = coordinates_train[all_samples_idx_to_keep]

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


            data_holder = DataHolder(x_train, y_train, bg_train, x_test, y_test, x_val, y_val, bg_val)
            # Fine tune the model
            fine_tuned_model = train_MLP(data_holder, config, base_model, auc_roc_val_all=training_results_val, auc_roc_test_all=training_results_test)

            # evaluate and log
            y_pred = predict(fine_tuned_model, x_test, device)
            aucs = evaluate_results(y_test, y_pred)
            df_aucs = pd.DataFrame(aucs, index=[species])
            df_results = pd.concat([df_results, df_aucs])

            # Save the training results
            if training_val_and_test_results_path is not None:
                log_training_results(run_type, args.training_val_and_test_results_path, training_results_val, region, species, "val")
                log_training_results(run_type, args.training_val_and_test_results_path, training_results_test, region, species, "test")

    #log_dir = f"{args.log_path}_seed_{hparams['seed_split']}"
    log_dir = args.log_path
    log_run(run_type, config, log_dir, df_results)
    
if __name__ == '__main__':
    main()