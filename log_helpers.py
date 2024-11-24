import pickle
from utils import RunType
import os
import pandas as pd
import wandb
import torch
import shutil

def log_run(run_type:RunType, config:dict, log_top_level_dir:str, results:dict):
    """Save the results and the configurations of the run to disk"""
    region = config['region']

    log_dir = _get_log_dir(run_type, region, log_top_level_dir)
    ensure_directory_structure(log_dir)

    log_results_name = _get_log_results_name()
    log_results_path = os.path.join(log_dir, log_results_name)
    df_results_log = pd.DataFrame(results)
    df_results_log.to_csv(log_results_path, index=True, mode="w")

    log_config_name = _get_log_config_name()
    log_config_path = os.path.join(log_dir, log_config_name)
    df_config = pd.DataFrame(config)
    df_config.to_csv(log_config_path, index=True, mode="w")


def ensure_directory_structure(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def read_log_run(run_type:RunType, region:str, log_top_level_dir:str):
    log_dir = _get_log_dir(run_type, region, log_top_level_dir)
    run_name = _get_log_results_name()
    log_path = os.path.join(log_dir, run_name)
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path, index_col=0)
    else:
        raise Exception(f'The logs for run type {run_type} and region {region} don\'t exist')
    return log_df


def _get_log_dir(run_type:RunType, region:str, log_top_level_dir:str):
    return os.path.join(log_top_level_dir, region, run_type.name)


def _get_log_results_name():
    return "results.csv"


def _get_log_config_name():
    return "config.csv"


# ==== wandb helpers ====

def initialize_wandb(run_type:RunType, config):
    if run_type == RunType.MULTI_SPECIES_MLP:
        proj_name = "multi_species"
        group_name = ""
        name = config['region']

    elif run_type == RunType.SINGLE_SPECIES_MLP:
        proj_name = "single_species"
        group_name = f"{config['region']}"
        name = f"{config['species']}"

    elif run_type == RunType.TRANSFER_LEARNING:
        proj_name = "transfer_learning"
        group_name = f"{config['region']}"
        name = f"{config['species']}"

    elif run_type == RunType.MAML:
        proj_name = "maml"
        group_name = f"{config['region']}"
        name = f"{config['species']}"
    else:
        AssertionError
    wandb.init(mode="disabled",project=proj_name, entity="meta-learn-sdm", group=group_name, name=name, config=config)

def save_files_to_wandb():
    wandb.save("train_model.py")
    wandb.save("training_helpers.py")
    wandb.save("losses.py")
    wandb.save("data_helpers.py")


# ==== model helpers ====

# Methods to save and retrieve existing models to and from disk

def _get_model_dir(model_base_dir, run_type:RunType, region:str):
    return os.path.join(model_base_dir, region, run_type.name)


def _get_model_name():
    return "model.pt"


def _get_model_path(model_base_dir, run_type:RunType, region:str):
    model_name = _get_model_name()
    model_dir = _get_model_dir(model_base_dir, run_type, region)
    return os.path.join(model_dir, model_name)


def save_model(model, model_base_dir, run_type:RunType, region):
    ensure_directory_structure(_get_model_dir(model_base_dir, run_type, region))
    model_path = _get_model_path(model_base_dir, run_type, region)
    torch.save(model, model_path)


# ==== training val and test result helpers ====
    
# Methods to save and retrieve the intermediate performance of the models on the validation set during training

def log_training_results(run_type:RunType, log_top_level_dir, training_results:list, region:str, species:str, val_or_test:str):

    log_dir = _get_training_results_dir(log_top_level_dir, region, run_type, species)
    ensure_directory_structure(log_dir)

    if val_or_test == "val":
        log_training_val_results_name = _get_training_val_results_name()
    elif val_or_test == "test":
        log_training_val_results_name = _get_training_test_results_name()
    log_results_path = os.path.join(log_dir, log_training_val_results_name)

    with open(log_results_path, 'wb') as file:
        pickle.dump(training_results, file)

# Read list to memory
def read_training_results(run_type, log_top_level_dir, region, species, val_or_test:str):
    
    log_dir = _get_training_results_dir(log_top_level_dir, region, run_type, species)
    
    if val_or_test == "val":
        log_training_val_results_name = _get_training_val_results_name()
    elif val_or_test == "test":
        log_training_val_results_name = _get_training_test_results_name()    
    
    log_results_path = os.path.join(log_dir, log_training_val_results_name)
    with open(log_results_path, 'rb') as file:
        training_val_results = pickle.load(file)
        return training_val_results
    

def _get_training_results_dir(log_path, region, run_type, species):
    return os.path.join(log_path, region, run_type.name, species)


def _get_training_val_results_name():
    return "training_val.txt"


def _get_training_test_results_name():
    return "training_test.txt"

