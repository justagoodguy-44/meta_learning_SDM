import os
import random
import numpy as np
import torch
import wandb
import learn2learn as l2l
import optuna
import copy

from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from tqdm import trange
from optuna.trial import TrialState

from data_helpers import SpeciesDataset, contains_positive_and_negative_samples, get_valid_species_indices_for_validation
from data_holder import DataHolder
from early_stopper import EarlyStopper
from optuna_data_holder import OptunaDataHolder
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from models import MLP
from losses import full_weighted_loss, get_species_weights, total_loss
from prg_corrected import create_prg_curve, calc_auprg
from utils import RunType, indices_to_one_hot, one_hot_to_indices


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy seed also uses by Scikit Learn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# ========================= MLP ==============================

def train_one_epoch_MLP(model, trainloader, loss_fn, species_weights, config, optimizer, scheduler):

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
        scheduler.step()

        return train_loss


def cross_validation_logic(model, x_val, y_val, bg_val, x_test, y_test, loss_fn, species_weights, valid_species_indices_for_validation, auc_roc_val_all, auc_roc_test_all, config, species_idx=None):
    
    if config["cross_validation_using_test"]:
        y_pred = predict(model, x_val, config["device"])
        auc_roc_val = evaluate_results(y_val, y_pred, species_idx)["auc_roc"][0]
    else:
        auc_roc_val = compute_metrics(model, x_val, y_val, bg_val, config, loss_fn, species_weights, valid_species_indices_for_validation, species_idx)["auc_roc"]
    
    if auc_roc_val_all is not None:
        auc_roc_val_all.append(auc_roc_val)
    if auc_roc_test_all is not None:
        y_pred = predict(model, x_test, config["device"])
        auc_roc_test = evaluate_results(y_test, y_pred, species_idx)["auc_roc"][0]
        auc_roc_test_all.append(auc_roc_test)
    return auc_roc_val

    
def train_MLP(data_holder:DataHolder, config:dict, base_model=None, verbose=False, auc_roc_val_all=None, auc_roc_test_all=None):
    """
    Args:
        data_holder (DataHolder): The data container
        config (dict): the configs
        base_model (torch model, optional): the base model to start from
        verbose (bool, optional): Adds some extra printing
        auc_roc_val_all (list, optional): Performance on validation set at every epoch is added to this list
        auc_roc_test_all (list, optional): Performance on test set at every epoch is added to this list

    Returns:
        The trained model
    """
    if config["cross_validation"]:
        # We compute the AUC only for species represented in both train and val sets and that have at least one positive sample in each
        valid_species_indices_for_validation = get_valid_species_indices_for_validation(data_holder.get("y_train"), data_holder.get("y_val"))

    data_holder.to_tensor(torch.float32, config["device"])
    x_train = data_holder.get("x_train")
    y_train = data_holder.get("y_train")
    bg_train = data_holder.get("bg_train")
    y_train = data_holder.get("y_train")
    x_test = data_holder.get("x_test")
    y_test = data_holder.get("y_test")
    if config["cross_validation"]:
        x_val = data_holder.get("x_val")
        y_val = data_holder.get("y_val")
        bg_val = data_holder.get("bg_val")        

    cross_validation = config["cross_validation"]
    early_stopping = config["early_stopping"]
    loss_fn = config["loss_fn"]
    
    species_weights = get_species_weights(y_train)
    species_weights = torch.tensor(species_weights, dtype=torch.float32).to(config["device"])

    trainset = SpeciesDataset(x_train, y_train, bg_train)
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)

    if early_stopping:
        early_stopper = EarlyStopper(increasing=True, patience=config["early_stopping_patience"])

    # check if a model is already passed as initializer, otherwise create model
    if base_model is None:
        model = MLP(input_size=x_train.shape[1], output_size=config["num_species"], num_layers=config["num_layers"], 
                    width=config["width_MLP"], dropout=config["dropout"])
        model.to(config["device"])  

    else:
        model = copy.deepcopy(base_model)        
        model.to(config["device"])  

        if cross_validation:
            # Can't test base model on all species with MAML since the model might have a smaller output sice than num_species
            if config["run_type"] != RunType.MAML:
                # Get performance of the base model
                auc_roc_val = cross_validation_logic(model, x_val, y_val, bg_val, x_test, y_test, loss_fn, species_weights, valid_species_indices_for_validation, auc_roc_val_all, auc_roc_test_all, config, config["species_idx"])
                if early_stopping:
                    early_stopper.add_new_score(auc_roc_val, model)
                if verbose:
                    print(f"Using base model with no fine tuning, val auc roc: {auc_roc_val}")
        
        adapt_model_from_multi_to_single_species_predictions(model, config)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_lambda = lambda epoch: config["learning_rate_decay"] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for i in trange(config["epochs"]):
        train_one_epoch_MLP(model, trainloader, loss_fn, species_weights, config, optimizer, scheduler)
        if cross_validation:
            auc_roc_val = cross_validation_logic(model, x_val, y_val, bg_val, x_test, y_test, loss_fn, species_weights, valid_species_indices_for_validation, auc_roc_val_all, auc_roc_test_all, config)
            if verbose:
                print(f"Epoch {i+1}, val auc roc: {auc_roc_val}")
            #Check for early stopping
            if not np.isnan(auc_roc_val) and early_stopping:
                early_stopper.add_new_score(auc_roc_val, model)
                if early_stopper.should_stop():
                    model = early_stopper.get_best_model()
                    if verbose:
                        print("Early stop")
                    break

    return model
    

# =============================================================
# ========================= MAML ==============================

def fast_adapt(batch, learner, loss_fn, adaptation_steps, shots, ways, total_classes_count, device, config):
        """Adapt the MAML model to specific task for adaption_steps steps"""

        x, y, bg = batch
        # Separate data into adaptation/evalutation sets
        adaptation_indices = np.zeros(x.size(0), dtype=bool)
        adaptation_indices[np.arange(shots*ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_x, adaptation_y, adaptation_bg = x[adaptation_indices], y[adaptation_indices], bg[adaptation_indices]
        evaluation_x, evaluation_y, evaluation_bg = x[evaluation_indices], y[evaluation_indices], bg[evaluation_indices]
        # Change back y from indices to one-hot for compatibility with loss calculation
        species_index = adaptation_y[0]
        adaptation_y = indices_to_one_hot(adaptation_y, total_classes_count).to(device)
        evaluation_y = indices_to_one_hot(evaluation_y, total_classes_count).to(device)
        # keep only column of the current task's species
        adaptation_y = adaptation_y[:, species_index]
        evaluation_y = evaluation_y[:, species_index]

        # Adapt the model
        for step in range(adaptation_steps):
            output = torch.sigmoid(learner(torch.cat((adaptation_x, adaptation_bg), 0)))[:, species_index]
            pred_adaptation_x = output[:len(adaptation_x)]
            pred_adaptation_bg = output[len(adaptation_x):]
            loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_adaptation_x, adaptation_y, pred_adaptation_bg)
            train_loss = total_loss(loss_dl_pos, loss_dl_neg, loss_rl, config["lambda_1"], config["lambda_2"], config["lambda_3"])
            learner.adapt(train_loss)
        # Evaluate the adapted model
        output = torch.sigmoid(learner(torch.cat((evaluation_x, evaluation_bg), 0)))[:, species_index]
        pred_evaluation_x = output[:len(evaluation_x)]
        pred_evaluation_bg = output[len(evaluation_x):]
        loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_evaluation_x, evaluation_y, pred_evaluation_bg)
        loss = total_loss(loss_dl_pos, loss_dl_neg, loss_rl, config["lambda_1"], config["lambda_2"], config["lambda_3"])
        return loss


def train_maml(data_holder:DataHolder, config, base_model=None):
    """
    Args:
        data_holder (DataHolder): The data container
        config (dict): the configs
        base_model (torch model, optional): the base model to start from

    Returns:
        The meta-trained model
    """

    # prepare data
    data_holder.to_tensor(torch.float32, config["device"])

    x_train = data_holder.get("x_train")
    y_train = data_holder.get("y_train")
    bg_train = data_holder.get("bg_train")
    y_train = data_holder.get("y_train")
    loss_fn = config["loss_fn"]

    species_weights = get_species_weights(y_train)
    species_weights = torch.tensor(species_weights, dtype=torch.float32).to(config["device"])

    total_species_count = y_train.shape[1]

    # Change y_train from one-hot to indices for compatibility with MetaDataset's implementation
    y_train = one_hot_to_indices(y_train).to(config["device"])
    bg_train = data_holder.get("bg_train")
    trainset = SpeciesDataset(x_train, y_train, bg_train)
    trainset = l2l.data.MetaDataset(trainset)

    # get transforms for learn2learn 
    train_transforms = [
        FusedNWaysKShots(trainset, n=config["ways"], k=2*config["shots"]),
        LoadData(trainset),
        RemapLabels(trainset),
        ConsecutiveLabels(trainset),
    ]
    # check if a model is already passed as initializer, otherwise create model
    if base_model is None:
        model = MLP(input_size=x_train.shape[1], output_size=config["num_species"], num_layers=config["num_layers"], 
                    width=config["width_MLP"], dropout=config["dropout"])
    else:
        model = copy.deepcopy(base_model)
    model.train()
    model.to(config["device"])  
    maml = l2l.algorithms.MAML(model, lr=config["fast_lr"], first_order=False)

    optimizer = torch.optim.SGD(maml.parameters(), lr=config["meta_lr"])
    loss_fn = config["loss_fn"]

    train_tasks = l2l.data.Taskset(
        trainset,
        task_transforms=train_transforms,
    )

    outer_loop_iters = config["outer_loop_iters"]
    tasks_batch_size = config["tasks_batch_size"]
    inner_loop_iters = config["inner_loop_iters"]
    shots = config["shots"]
    ways = config["ways"]

    for iteration in trange(outer_loop_iters):
        optimizer.zero_grad()
        for task in trange(tasks_batch_size):
            learner = maml.clone()
            batch = train_tasks.sample()
            loss = fast_adapt(batch, learner, loss_fn, inner_loop_iters, shots, ways, total_species_count, config["device"], config)
            loss.backward()

        # Average the accumulated gradients and optimize
        for param in maml.parameters():
            param.grad.data.mul_(1.0 / tasks_batch_size)
        optimizer.step()

    return model


    

# =============================================================
# ========================= EVALUATION ==============================

def predict(model, x_eval, device):
    model.eval()
    y_pred = torch.sigmoid(model(torch.tensor(x_eval, dtype=torch.float32).to(device))).to("cpu")
    model.train()
    return y_pred


def evaluate_results(y_truth, y_pred, species_idx=None):
    """_summary_
    Get stats and results based on the predictions and ground truth
    Args:
        y_truth: the ground truth
        y_pred: the predictions
        species_idx: in the multi species case this param allows to get the performance on a specific species

    Returns:
        dict: areas under curve for roc and pr
    """

    # adjust types
    if torch.is_tensor(y_truth):
        y_truth = y_truth.detach().to("cpu").numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().to("cpu").numpy()

    # adjust dimensions    
    if y_truth.ndim == 1:
        y_truth = y_truth[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    # In case of multi species predictions on a specific species
    if species_idx is not None and y_pred.shape[1] != 1:
        y_pred = (y_pred[:, species_idx])[:, np.newaxis]

    assert(y_pred.shape == y_truth.shape), f"shapes dont match! y_pred has shape {y_pred.shape} while y_truth has shape {y_truth.shape}"

    non_nan_elements = np.logical_not(np.isnan(y_truth)) # NaN elements correspond to other groups of species 
    eval_auc_rocs = []
    eval_auc_prgs = []
    species_count = y_truth.shape[1]
    for i in range(species_count):
        y_truth_col = y_truth[:, i]
        y_truth_col = y_truth_col[non_nan_elements[:, i]]
        if contains_positive_and_negative_samples(y_truth_col):    
            y_pred_col = y_pred[:, i]
            y_pred_col = y_pred_col[non_nan_elements[:, i]]      
            eval_auc_rocs.append(roc_auc_score(y_truth_col, y_pred_col))
            prg_curve = create_prg_curve(y_truth_col, y_pred_col)
            eval_auc_prgs.append(calc_auprg(prg_curve))
    mean_eval_auc_roc = np.mean(eval_auc_rocs)
    mean_eval_auc_prgs = np.mean(eval_auc_prgs)
    aucs = {"auc_roc":[mean_eval_auc_roc], "auc_prgs":[mean_eval_auc_prgs]}

    return aucs


def compute_metrics(model, x, y, bg, config, loss_fn, species_weights, indices_non_zeros_samples, species_idx=None):
    """Compute some general metrics about the model's current predictions on x"""

    model.eval()
    pred = torch.sigmoid(model(torch.cat((x, bg[:len(x)]), 0)))
    model.train()
    # for evaluating multi species on a specific species
    if species_idx is not None:
        pred = (pred[:, species_idx])[:, np.newaxis]
    pred_x = pred[:len(x)]
    pred_bg = pred[len(x):]

    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    
    # Loss function in pytorch
    loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_x, y, pred_bg, species_weights)
    loss_dl_pos, loss_dl_neg, loss_rl = loss_dl_pos.item(), loss_dl_neg.item(), loss_rl.item()
    if len(pred_bg) == 0:
        loss_rl = 0
    loss = config["lambda_1"] * loss_dl_pos + config["lambda_2"] * loss_dl_neg + config["lambda_3"] * loss_rl
    
    # Metrics in numpy
    pred = pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    
    y_with_bg = np.concatenate((y, np.zeros((len(pred_bg), y.shape[1]))), axis=0)
    # check if there are no presences in the validation set, as is bound to happen when the number of total presences is extremely small
    
    if len(indices_non_zeros_samples)==0:
        # return nan because rocs and prgs are undefined for single class
        mean_auc_roc= np.nan
        mean_auc_prg = np.nan
    else:
        auc_rocs = roc_auc_score(y_with_bg[:, indices_non_zeros_samples], pred[:, indices_non_zeros_samples], average=None)
        auc_prgs = []
        for i in range(len(y_with_bg[0])):
            if i in indices_non_zeros_samples:
                prg_curve = create_prg_curve(y_with_bg[:, i], pred[:, i])
                auc_prgs.append(calc_auprg(prg_curve))
        mean_auc_roc = np.mean(auc_rocs)
        mean_auc_prg = np.mean(auc_prgs)

    results = {
        "loss_dl_pos": loss_dl_pos,
        "loss_dl_neg": loss_dl_neg,
        "loss_rl": loss_rl,
        "loss": loss,
        "auc_roc": mean_auc_roc,
        "auc_prg": mean_auc_prg,
    }

    return results


def adapt_model_from_multi_to_single_species_predictions(model, config):
    model.change_output_size(1)
    if config["only_fine_tune_last_layer"]:
        model.freeze_all_but_last_layer()
    model.change_dropout(config["dropout"])
    model.to(config["device"])