# Applying meta-learning to SDMs

Welcome to this project! I will try to explain the main aspects to navigate and understand the codebase

## Project objectives

The goal of this project was to implement an SDM (species distribution model) using different machine learning techniques and analyze the advantages and setbacks of each one. These different techniques were :
- **Multi-species**: Using an MLP with the last layer of size equal to the number of species we wish to predict for and feeding it all species' data.
- **Single-species**: Using an MLP with the last layer of size 1, each species would have its own model trained exclusively on the data of that species.
- **Transfer-learning**: We would first start by training a multi-species base model, then each species would have its own model by fine-tuning this base model exclusively on the data of that species.
- **MAML**: Basing ourselves on the proposed [MAML](https://arxiv.org/abs/1703.03400) method, an original model would be meta-learned on samples of all species, then each species would have its own model by fine-tuning this base model exclusively on the data of that species. 

## Navigating the files and folders
### Folders:
- **data**: contains the data used to train and test the models.
- **hparams**: contains the different hyperparameter files.
- **logs**: contains the results and configurations of the different runs.
- **models**: contains the saved multi-species pytorch models.
- **Papers**: contains different research papers related to this project
- **Scalers**: contains the scalers used to scale the data.
- **training_val_and_test_results**: contains logs of the evolution of the performance on validation and test set of some runs during training.

### Files:
- **data_exploration.ipynb**: a notebook to have some first ideas and analysis on the dataset we are working with.
- **data_helpers.py**: methods to help retrieve and manipulate the data.
- **data_holder.py**: a convenience class to bundle data.
- **early_stopper.py**: implements everything needed to use early stopping during model training.
- **hparams_generator.ipynb**: a notebook to generate hyperparameter files for different runs.
- **log_helpers.py**: methods to help save and load logs and models.
-  **losses.py**: implements the necessary functions to calculate losses that also take species' weights into account.
-  **main_runner.ipynb**: a notebook used to launch any run.
-  **model_performance_analysis.ipynb**: a notebook to view the results of the different runs, as well as some more advanced data visualization.
-  **models.py**: defines the MLP and some useful methods for it.
-  **prg_corrected.py**: defines useful methods creating prg (Precision-Recall-Gain) curves.
-  **requirements.txt**: file to help download the necessary python packages all at once.
-  **train_maml.py**: the main file to train and evaluate a MAML model.
-  **train_mlp_multi_species_model.py**: the main file to train and evaluate a multi-species model.
-  **train_mlp_single_species_model.py**: the main file to train and evaluate a single-species model.
-  **train_transfer_learning.py**: the main file to train and evaluate a transfer-learning model.
-  **training_helpers.py**: helper methods to train a model
-  **utils.py**: miscelleanous useful methods