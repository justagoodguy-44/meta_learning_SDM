import random

import pandas as pd
import numpy as np
import verde as vd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


elith_data_dir = 'data/elith_2020/Records/'
valavi_bg_dir = 'data/elith_2020/Valavi_DataS1/background_50k/'
rasters_dir = 'data/elith_2020/Environment'

region_list = ['AWT', 'CAN', 'NSW', 'NZ', 'SA', 'SWI']
group_dictionary = {
    'AWT': ['bird', 'plant'],
    'CAN': [],
    'NSW': ['ba', 'db', 'nb', 'ot', 'ou', 'rt', 'ru', 'sr'],
    'NZ': [],
    'SA': [],
    'SWI': []
}

# Covariates of Elith et al. 2020
covariate_dictionary = {
    'AWT': ['bc01', 'bc04', 'bc05', 'bc06', 'bc12', 'bc15', 'bc17', 'bc20', 'bc31', 'bc33', 'slope', 'topo', 'tri'],
    'CAN': ['alt', 'asp2', 'ontprec', 'ontprec4', 'ontprecsd', 'ontslp', 'onttemp', 'onttempsd', 'onttmin4', 'ontveg', 'watdist'],
    'NSW': ['cti', 'disturb', 'mi', 'rainann', 'raindq', 'rugged', 'soildepth', 'soilfert', 'solrad', 'tempann', 'tempmin', 'topo', 'vegsys'],
    'NZ': ['age', 'deficit', 'dem', 'hillshade', 'mas', 'mat', 'r2pet', 'rain', 'slope', 'sseas', 'toxicats', 'tseas', 'vpd'],
    'SA': ['sabio1', 'sabio2', 'sabio4', 'sabio5', 'sabio6', 'sabio7', 'sabio12', 'sabio15', 'sabio17', 'sabio18',],
    'SWI': ['bcc', 'calc', 'ccc', 'ddeg', 'nutri', 'pday', 'precyy', 'sfroyy', 'slope', 'sradyy', 'swb', 'tavecc', 'topo']
}

# Covariates of Valavi et al. 2022
valavi_covariate_dictionary = {
    'AWT': ['bc04', 'bc05', 'bc06', 'bc12', 'bc15', 'slope', 'topo', 'tri'],
    'CAN': ['alt', 'asp2', 'ontprec', 'ontslp', 'onttemp', 'ontveg', 'watdist'],
    'NSW': ['cti', 'disturb', 'mi', 'rainann', 'raindq', 'rugged', 'soildepth', 'soilfert', 'solrad', 'tempann', 'topo', 'vegsys'],
    'NZ': ['age', 'deficit', 'hillshade', 'mas', 'mat', 'r2pet', 'slope', 'sseas', 'toxicats', 'tseas', 'vpd'],
    'SA': ['sabio12', 'sabio15', 'sabio17', 'sabio18', 'sabio2', 'sabio4', 'sabio5', 'sabio6'],
    'SWI': ['bcc', 'calc', 'ccc', 'ddeg', 'nutri', 'pday', 'precyy', 'sfroyy', 'slope', 'sradyy', 'swb', 'topo']
}

# Same for Valavi et al. 2022 and Elith et al. 2020
categorical_covariates = { 
    'AWT': [],
    'CAN': ['ontveg'], 
    'NSW': ['vegsys'], 
    'NZ': [], #'age', 'toxicats' are ordinal variables -> no one hot encoding
    'SA': [],
    'SWI': [] #'calc' is binary, no need to one hot encode
}


class SpeciesDataset(Dataset):

    def __init__(self, x, y, bg):
        self.x = x
        self.y = y
        self.bg = bg
        self.length = len(self.x)
        self.num_bg = len(self.bg)
    
    def __getitem__(self, idx):
        idx_bg = random.randint(0, self.num_bg - 1)
        return self.x[idx], self.y[idx], self.bg[idx_bg]
    
    def __len__(self):
        return self.length
    
    
def get_data_one_region(region, co_occurrence=True, valavi=False):
    """
    Returns:
        tuple of ndarrays corresponding to the data of this region
    """
    covs = get_covariates(region, valavi=valavi)
    cat_covs = categorical_covariates[region]
    
    # Get presence-only occurrence records
    train = pd.read_csv(elith_data_dir + 'train_po/' + region + 'train_po.csv')
    train = train[['spid'] + covs + ["x", "y"]].reset_index(drop=True)
    
    # One hot encoding of categorical variables to obtain x_train (covariates)
    x_train = pd.get_dummies(train, columns=cat_covs).drop(["spid"], axis=1)
    if co_occurrence:
        # Merge species at same location (same covariates by definition)
        x_train = x_train.groupby(["x", "y"]).mean().reset_index()
    coordinates_train = x_train[["x", "y"]].to_numpy()
    x_train = x_train.drop(["x", "y"], axis=1)
    x_train = x_train.to_numpy()
    
    # Encode the presence into a binary vector to obtain y_train
    y_train = pd.get_dummies(train, columns=['spid']).drop(covs, axis=1)
    if co_occurrence:
        # Merge species at same location
        y_train = y_train.groupby(["x", "y"]).sum().reset_index()
    y_train = y_train.drop(["x", "y"], axis=1)
    y_train = y_train.to_numpy().clip(0, 1)
    
    # Get background points
    bg = pd.read_csv(valavi_bg_dir + region + '.csv')
    bg = pd.get_dummies(bg[covs], columns=cat_covs).to_numpy()
    
    # Presence-absence and covariates
    groups = group_dictionary[region]
    if len(groups) > 0:
        test_pa = []
        test_env = []
        for group in groups:
            test_pa.append(pd.read_csv(elith_data_dir + 'test_pa/' + region + 'test_pa_' + group + '.csv'))
            test_env.append(pd.read_csv(elith_data_dir + 'test_env/' + region + 'test_env_' + group + '.csv'))
        test_pa = pd.concat(test_pa)
        test_env = pd.concat(test_env)
    else:
        test_pa = pd.read_csv(elith_data_dir + 'test_pa/' + region + 'test_pa.csv')
        test_env = pd.read_csv(elith_data_dir + 'test_env/' + region + 'test_env.csv')
    x_test = pd.get_dummies(test_env.sort_values('siteid')[covs], columns=cat_covs).to_numpy()
    coordinates_test = test_env.sort_values('siteid')[["x", "y"]].to_numpy()
    y_test = np.array(test_pa.sort_values('siteid')[get_species_list(region, remove=False)])

    
    return x_train, y_train, coordinates_train, x_test, y_test, coordinates_test, bg
    

def get_data_single_species(region, species, co_occurrence=True, valavi=False):
    """
    Returns:
        tuple of ndarrays corresponding to this species
    """
    # Get data for all species, keeping all covariates, the unwanted will be dropped later
    x_train_multi, y_train_multi, coordinates_train_multi, x_test_multi, y_test_multi, coordinates_test_multi, bg_multi = get_data_one_region(region, co_occurrence, valavi)
    # Get the indices where the specific species appears
    region_species_list = np.array(get_species_list(region, remove=False))
    species_one_hot = region_species_list==np.array(species)

    # Check that the species actually exists in this region
    if np.max(species_one_hot)==0:
        raise Exception('The provided species does not exist in the provided region')
    species_idx = np.argmax(region_species_list==np.array(species))

    # Keep only the ground-truth column of the specific species column for ground-truth
    x_train = x_train_multi
    y_train = (y_train_multi[:, species_idx])[:,np.newaxis]
    coordinates_train = coordinates_train_multi
    x_test = x_test_multi
    y_test = (y_test_multi[:,species_idx])[:,np.newaxis]
    coordinates_test = coordinates_test_multi
    bg = bg_multi

    return x_train, y_train, coordinates_train, x_test, y_test, coordinates_test, bg


def one_hot_covariates(x, region, valavi=False):
    
    covs = get_covariates(region, valavi=valavi)
    cat_covs = categorical_covariates[region]
    
    x = pd.DataFrame(x, columns=covs)
    x = pd.get_dummies(x, columns=cat_covs).to_numpy()
    
    return x


def get_species_list(region, remove=False):
    """
    Returns the list of species for a given region
    Args:
        region (string): the region name
        remove (bool, optional): If true then the species nsw30 which has only 2 occurrences in train is not returned in the list

    Returns:
        list: the list of species
    """
    species = list(pd.read_csv(elith_data_dir + 'train_po/' + region + 'train_po.csv')['spid'].unique())
    if region == 'NSW' and remove: 
        species.remove('nsw30') # species with only 2 occurrences in train
    return species


def get_covariates(region, valavi=True):
    """Returns the list of covariates for a given region."""
    
    if valavi:
        covs = valavi_covariate_dictionary[region]
    else:
        covs = covariate_dictionary[region]
    
    return covs

def get_region_list():
    return region_list


def split_train_data_for_cross_validation(x_train, y_train, coordinates_train, bg, hparams):
    """Generates the splits for cross validation by using part of the train (presence-only) data"""
    if hparams["cross_validation"]:
        if hparams["blocked_cv"]:
            kfold = vd.BlockKFold(shape=hparams["num_cv_blocks"], n_splits=int(1/hparams["val_size"]), shuffle=True, balance=True).split(coordinates_train)
            ind_train, ind_val = next(kfold)
            x_train, x_val, y_train, y_val = x_train[ind_train], x_train[ind_val], y_train[ind_train], y_train[ind_val]
        else:
            if y_train.ndim == 1 or y_train.shape[1]==1:
                presences_idx = np.where(y_train == 1)[0]
                absences_idx = np.ones(presences_idx.shape, dtype=int) - presences_idx
                x_train_presences, x_val_presences, y_not_val_presences, y_val_presences = train_test_split(x_train[presences_idx], y_train[presences_idx], test_size=hparams["val_size"])
                x_train_absences, x_val_absences, y_not_val_absences, y_val_absences = train_test_split(x_train[absences_idx], y_train[absences_idx], test_size=hparams["val_size"])  
                x_train = np.concatenate((x_train_presences, x_train_absences), axis=0)
                x_val = np.concatenate((x_val_presences, x_val_absences), axis=0)
                y_train = np.concatenate((y_not_val_presences, y_not_val_absences), axis=0)
                y_val = np.concatenate((y_val_presences, y_val_absences), axis=0)
            else:
                x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=hparams["val_size"])
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=hparams["val_size"])     
        # Add background points to validation set
        num_presences_val = y_val.sum()
        num_bg_val = num_presences_val
        np.random.shuffle(bg)
        bg_train = bg[num_bg_val:]
        bg_val = bg[:num_bg_val] 
    else:
        bg_train = bg
        x_val = None
        y_val = None
        bg_val = None
    
    return x_train, y_train, x_val, y_val, bg_train, bg_val


def split_test_data_for_cross_validation(x_test, y_test, coordinates_test, hparams):
    """_summary_
    Generates the splits for cross validation by using part of the test (presence-absence) data

    """

    if hparams["cross_validation"]:
        if hparams["blocked_cv"]:
            kfold = vd.BlockKFold(shape=hparams["num_cv_blocks"], n_splits=int(1/hparams["val_size"]), shuffle=True, balance=True).split(coordinates_test)
            ind_train, ind_val = next(kfold)
            x_not_val, x_val, y_not_val, y_val = x_test[ind_train], x_test[ind_val], y_test[ind_train], y_test[ind_val]
        else:
            x_not_val, x_val, y_not_val, y_val = train_test_split(x_test, y_test, test_size=hparams["val_size"])     

    else:
        x_not_val = x_test
        y_not_val = y_test
        x_val = None
        y_val = None

    return x_not_val, y_not_val, x_val, y_val 



def keep_only_species_with_enough_samples(labels, min_samples):
    """
    Returns the indices of the rows corresponding to species with more than num_samples samples
     Args:
        labels (nparray): the one-hot labels
        num_samples (int): the minimum number of samples
    """

    column_sums = np.sum(labels, axis=0)
    column_indices_above_threshold = np.where(column_sums >= min_samples)[0]
    row_indices_above_threshold = np.any(labels[:, column_indices_above_threshold] > 0, axis=1)
    row_indices_above_threshold = np.where(row_indices_above_threshold==True)[0]

    return row_indices_above_threshold.tolist(), column_indices_above_threshold.tolist()


def contains_positive_and_negative_samples(labels):
    if labels.ndim > 1:
        if labels.shape[1] > 1:
            # Doesn't make sense if there is more than one species in the labels
            raise AssertionError
        
        contains_negative = 0 in labels[:,0]
        contains_positive = 1 in labels[:,0]
    else:
        contains_negative = 0 in labels
        contains_positive = 1 in labels
    return contains_negative and contains_positive


def get_valid_species_indices_for_validation(y_train, y_val):
    """
    Returns:
        ndarray: array of column indices in which the species have at least one presence in the train and val set
    """    
    valid_indices = np.intersect1d(np.sum(y_val, axis=0).nonzero(), 
                                                    np.sum(y_train, axis=0).nonzero())
    return valid_indices


def get_dict_from_species_name_to_nb_of_samples(region):
    region_species_names = get_species_list(region)
    species_to_samples = {}
    for species in region_species_names:
        species_data = get_data_single_species(region, species)[1]
        species_to_samples[species] = np.sum(species_data)
    return species_to_samples


def get_species_idx_from_name(species:str, region:str):
    region_species_names = get_species_list(region)
    return region_species_names.index(species)


def get_species_samples_idx(labels, species_idx=None):
    """_summary_
    Returns a list of indices of a species' samples
    Args:
        labels (ndarray): the labels, of size (samples), (samples,1) or (samples, num_species) if the shape is (samples, num_species) then species_idx will specify the column to examine
        species_idx (int): In case of multi species it specifies the index of the species to look at

    """
    idx_list = []
    if labels.ndim == 1:
        labels = labels[:, np.newaxis]
    
    if labels.shape[1] == 1:
        species_idx = 0

    for i in range(labels.shape[0]):
        if labels[i, species_idx] == 1:
            idx_list.append(i)
    
    return idx_list