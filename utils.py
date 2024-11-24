from enum import Enum
import argparse
import json
import torch
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np

from data_helpers import get_data_single_species, get_species_list



class RunType(Enum):
    MULTI_SPECIES_MLP = 1
    SINGLE_SPECIES_MLP = 2
    TRANSFER_LEARNING = 3
    MAML = 4

def get_mlp_arg_parser():    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--hparams_path", type=str, required=True, help="path of the params file")
    arg_parser.add_argument("--log_path", type=str, required=True, help="Path to directory where the logs will be stored")
    arg_parser.add_argument("--region", type=str, help="region of interest", default="SWI")
    arg_parser.add_argument("--run_name", type=str, help="name of the run", default="unspecified_run")
    arg_parser.add_argument("--scaler_path", type=str, help="path of the standard scaler", default=None)
    arg_parser.add_argument("--save_model_path", type=str, help="path to directory where the models will be stored, default None", default=None)
    arg_parser.add_argument("--training_val_and_test_results_path", type=str, help="path to directory where the results will be stored, default None", default=None)
    return arg_parser


def get_transfer_learning_arg_parser():
    arg_parser = get_mlp_arg_parser()
    arg_parser.add_argument("--load_model_path", type=str, help="the path to the model that will be used to initialize the params, if none the model will be generated", default=None)
    arg_parser.add_argument("--only_last_layer", type=bool, help="if true only the last layer will be fine-tuned, otherwise the whole model", default=True)
    return arg_parser


def get_meta_learning_arg_parser():
    arg_parser = get_mlp_arg_parser()
    arg_parser.add_argument("--load_model_path", type=str, help="the path to the model that will be used as the base for meta-testing", default=None)
    arg_parser.add_argument("--only_last_layer", type=bool, help="if true only the last layer will be fine-tuned, otherwise the whole model", default=True)
    arg_parser.add_argument("--specific_species", type=str, help="the name of a specific species if you only want to fine tune on this species and not all those in the region", default=None)
    arg_parser.add_argument("--num_samples", type=int, help="if specific_species is not none, this many of its samples will be kept for training, the others will be discarded", default=None)
    
    return arg_parser


def write_params_to_json(config:dict, path):
    with open(path, 'w') as file:
        json.dump(config, file, indent=4)


def read_params_from_json(path) -> dict:
    with open(path, 'r') as file:
        config = json.load(file)
        return config


def one_hot_to_indices(input_tensor):
    return torch.argmax(input_tensor, dim=1)


def indices_to_one_hot(indices, num_classes):
    one_hot_tensor = torch.zeros((len(indices), num_classes))
    one_hot_tensor.scatter_(1, indices.view(-1, 1), 1)
    return one_hot_tensor


def scale_data(x_train, x_test, bg, scaler_path):
    if scaler_path is not None:
            sc = load(scaler_path)
    else:
        sc = StandardScaler().fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    bg = sc.transform(bg)
    return x_train, x_test, bg