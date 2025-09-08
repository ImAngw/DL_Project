import numpy as np
import random
import torch
from my_custom_ai.utils.save_score_utils import WeightsAndBiases



class Config:
    """
    Represents the configuration for an experiment with validation of input parameters.

    This class allows the user to set up configurations for an experiment, such as the
    name of the experiment, directory for checkpoint storage, batch size, number of
    epochs, logging type, and more. The class includes validation for input parameters
    and initializes a logger if required.

    :ivar exp_name: Name of the experiment.
    :type exp_name: str
    :ivar checkpoint_dir: Directory where the model checkpoints will be saved.
    :type checkpoint_dir: str
    :ivar device: Device to be used for training/testing (e.g., 'cpu', 'cuda').
    :type device: str
    :ivar require_early_stop: Boolean flag indicating whether early stopping is required.
    :type require_early_stop: bool
    :ivar early_stopping_patience: Number of epochs to wait for improvement before stopping early.
    :type early_stopping_patience: int
    :ivar early_stopping_min_delta: Minimum delta value for improvement to avoid early stopping.
    :type early_stopping_min_delta: float
    :ivar seed: Seed value for reproducibility.
    :type seed: int
    :ivar num_workers: Number of workers for data loading.
    :type num_workers: int
    :ivar batch_size: Batch size for training/testing.
    :type batch_size: int
    :ivar num_epochs: Number of epochs for the training.
    :type num_epochs: int
    :ivar logger: Logger object initialized for the experiment.
    :type logger: WeightsAndBiases or None
    """
    _ALLOWED_LOGGERS = ['wandb']

    def __init__(self,
                 experiment_name,
                 checkpoint_dir,

                 batch_size=32,
                 num_epochs=50,
                 lr=1e-4,
                 num_workers=2,
                 seed=42,
                 device='cpu',

                 require_early_stop=False,
                 early_stop_desc=True,
                 early_stopping_patience=10,
                 early_stopping_min_delta=0.002,

                 logger_type='wandb',
                 logger_init=None,
                 logger_update_at_each_epoch=True,
                 ):

        self.exp_name = experiment_name                           # Name of the experiment
        self.checkpoint_dir = checkpoint_dir                      # Directory where the model will be saved
        self.device = device
        self.require_early_stop = require_early_stop
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stop_desc = early_stop_desc
        self.seed = seed
        set_seeds(seed)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.logger = self._return_logger_obj(logger_type, logger_init)     # Logger initialization
        self.logger_update_at_each_epoch = logger_update_at_each_epoch    # if False, it will be updated at each train step

        # Validation on each variable
        self._validate()


    def __repr__(self):
        return (f"<Config(experiment_name={self.exp_name}, checkpoint_dir={self.checkpoint_dir}, "
                f"batch_size={self.batch_size}, num_epochs={self.num_epochs},"
                f"logger={self.logger}, num_workers={self.num_workers}, "
                f"seed={self.seed}, device={self.device}, require_early_stop={self.require_early_stop}, "
                f"early_stopping_patience={self.early_stopping_patience}, "
                f"early_stopping_min_delta={self.early_stopping_min_delta})>")

    def _validate(self):
        self._name_validation()
        self._checkpoint_validation()
        self._pos_number_validation(self.batch_size, "batch_size", int)
        self._pos_number_validation(self.num_workers, "num_workers", int)
        self._pos_number_validation(self.seed, "seed", int)
        self._pos_number_validation(self.num_epochs, "num_epochs",int)
        self._pos_number_validation(self.early_stopping_patience, "early_stopping_patience", int)
        self._pos_number_validation(self.early_stopping_min_delta, "early_stopping_min_delta", float)

    def _name_validation(self):
        if not self.exp_name:
            raise ValueError("Experiment name is required.")

    def _checkpoint_validation(self):
        """
        Validates the presence of a checkpoint directory.
        checkpoint_dir: directory where the model will be saved.

        :raises ValueError: Raised when ``checkpoint_dir`` is not set or is None.

        """
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory is required.")


    @staticmethod
    def _pos_number_validation(value, label, num_type):
        if type(value) != num_type:
            raise ValueError(f"{label} must be {num_type}, got {type(value)} instead.")
        else:
            if value <= 0:
                raise ValueError(f"{label} must be greater than zero, got {value} instead.")


    @staticmethod
    def _return_logger_obj(logger_type, logger_init):
        if logger_type not in Config._ALLOWED_LOGGERS:
            raise ValueError(f"Logger must be one of {Config._ALLOWED_LOGGERS}, got {logger_type} instead.")

        if logger_init is not None:
            if logger_type == 'wandb':
                return WeightsAndBiases(logger_init)

        return None



def N(x: torch.Tensor):
    """Get pure value"""
    # detach from a computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()

def set_seeds(seed):
    """Set seeds for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
