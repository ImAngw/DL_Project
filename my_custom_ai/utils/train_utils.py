import torch
from my_custom_ai.utils.misc_utils import Config
import os
from abc import abstractmethod, ABC




def save_model(configs:Config, model, verbose=False, f_name=None):
    """Save the model"""
    if not f_name:
        f_name = configs.exp_name

    os.makedirs(configs.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(configs.checkpoint_dir, f_name + '.pth')

    torch.save(model.state_dict(), str(model_path))
    if verbose:
        print(f"Saved model at path={model_path}")

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize the AverageMeter with default values."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def reset(self):
        """Reset all statistics to zero."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        """Update statistics with a new value.
        Args:
            val: The value to update with
            n: Weight of the value (default: 1)
        """
        # update statistic with a given new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping strategy"""

    def __init__(self, configs:Config, verbose=False):

        self._configs = configs
        self.patience = configs.early_stopping_patience
        self.min_delta = configs.early_stopping_min_delta
        self.descending = configs.early_stop_desc
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose


    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            # initialize best score
            self.best_score = score
            self.checkpoint(model)  # first model
        else:
            if self.descending:
                if score < self.best_score + self.min_delta:
                    # no improvement seen
                    self.counter += 1
                    print(f"Early stopping counter={self.counter} out of patience={self.patience}")

                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    # we see an improvement
                    self.best_score = score
                    self.checkpoint(model)
                    self.counter = 0
            else:
                if score > self.best_score + self.min_delta:
                    # no improvement seen
                    self.counter += 1
                    print(f"Early stopping counter={self.counter} out of patience={self.patience}")

                    if self.counter >= self.patience:
                        self.early_stop = True

                else:
                    # we see an improvement
                    self.best_score = score
                    self.checkpoint(model)
                    self.counter = 0

        '''elif score < self.best_score + self.min_delta:
            # no improvement seen
            self.counter += 1
            print(f"Early stopping counter={self.counter} out of patience={self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            # we see an improvement
            self.best_score = score
            self.checkpoint(model)
            self.counter = 0'''

    def checkpoint(self, model):
        """Save the best model"""
        if self.verbose:
            print(f"Updated best_score={self.best_score:.3f}")

        save_model(self._configs, model, self.verbose)


class FunctionContainer(ABC):
    """
    Defines an abstract base class for containers that manage batch processing, loss computation,
    and performance evaluation for use in machine learning workflows.

    This class serves as a blueprint for concrete implementations, enforcing the structure and
    methodology. It ensures that derived classes implement the required methods for batch
    processing, loss computation, and performance evaluation.

    :ivar self.batch_extractor: Abstract method to process and extract useful components from a data batch.
    :type self.batch_extractor: Callable
    :ivar self.loss_function: Abstract method to compute scalar loss from model output and target labels.
    :type self.loss_function: Callable
    :ivar self.performance_score: Abstract method to calculate a performance score, such as accuracy,
      from model output and target labels.
    :type self.performance_score: Callable
    """

    @abstractmethod
    def batch_extractor(self, batch, *args, **kwargs):
        """
        It must take a batch (in whatever form) and returns the batch and the y (labels or whatever) in a form useful for loss_function.
        Expected return: elaborated_batch, y.
        NOTE: If the model requires multiple inputs, it must return {batch_name: elaborated_batch, ...}, y
        """
        pass

    @abstractmethod
    def loss_function(self, model_output, y, *args, **kwargs):
        """It takes the model output (on a batch) and y and returns the scalar loss."""
        pass

    @abstractmethod
    def validation_performance(self, model, loader, *args, **kwargs):
        """It takes the model and validation loader and returns the scalar performance (accuracy or whatever)."""
        pass

    @abstractmethod
    def test_performance(self, model, loader, pbar, *args, **kwargs):
        """It takes the model and test loader and returns the scalar performance (accuracy or whatever).
            Remember to call pbar.update(1) at the end of each test loop!"""
        pass


class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, metric=None):
        try:
            self.scheduler.step(metric)
        except TypeError:
            self.scheduler.step()
