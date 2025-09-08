from my_custom_ai.utils.train_utils  import AverageMeter, SchedulerWrapper, FunctionContainer, EarlyStopping
from tqdm import tqdm
import sys
import torch
from my_custom_ai.utils.misc_utils import Config, N



class CustomTraining:
    """
    CustomTraining is a class designed to manage and facilitate training processes for machine learning models. It
    handles training loops, validation, performance monitoring, and supports additional functionality like optimization
    scheduling and early stopping. This class is modular and configurable, allowing for various use cases such as
    classification tasks. It integrates seamlessly with a provided configuration system, making it extensible and flexible.

    CustomTraining also supports advanced features such as scheduler updates per batch or per epoch, a logging interface
    for external systems, and a container for custom loss functions and performance evaluation metrics.

    :ivar configs: Configuration object that governs the training setup, including parameters like epochs, early stopping,
        and logging.
    :type configs: Config
    :ivar model: Model to be trained.
    :ivar optimizer: Optimization algorithm instance for updating model parameters.
    :ivar train_loader: DataLoader object for fetching training data batches.
    :ivar val_loader: DataLoader object for validation data.
    :ivar test_loader: DataLoader object for test data.
    :ivar function_container: A collection of custom functions for operations like batch extraction and loss computation.
    :type function_container: FunctionContainer
    :ivar loss_meter: A utility to track and update average loss over iterations.
    :type loss_meter: AverageMeter
    :ivar performance_meter: A utility to track and update average performance metrics like accuracy.
    :type performance_meter: AverageMeter
    :ivar scheduler: Learning rate scheduler for adjusting optimization parameters dynamically.
    :ivar scheduler_metric: Metric used for guiding scheduler updates, if applicable.
    :ivar step_scheduler_each_batch: Boolean flag determining whether to step the scheduler after each batch or epoch.
    :type step_scheduler_each_batch: bool
    :ivar verbose: Boolean flag to enable detailed logs during training.
    :type verbose: bool
    :ivar args_l: Positional arguments for the loss function.
    :type args_l: tuple
    :ivar kwargs_l: Keyword arguments for the loss function.
    :type kwargs_l: dict
    :ivar args_b: Positional arguments for the batch extraction function.
    :type args_b: tuple
    :ivar kwargs_b: Keyword arguments for the batch extraction function.
    :type kwargs_b: dict
    :ivar args_a: Positional arguments for the performance evaluation function.
    :type args_a: tuple
    :ivar kwargs_a: Keyword arguments for the performance evaluation function.
    :type kwargs_a: dict
    :ivar grad_clip_norm: If set, clips gradient norm to this value before optimizer step.
    :type grad_clip_norm: float | None
    :ivar use_amp: Enable CUDA Automatic Mixed Precision training for stability/perf.
    :type use_amp: bool
    :ivar grad_accum_steps: Accumulate gradients over this many steps before optimizer step.
    :type grad_accum_steps: int
    """

    def __init__(self, configs:Config, model, function_container:FunctionContainer, optimizer, train_loader,
                 val_loader=None, test_loader=None, eval_on_validation=False, scheduler=None, scheduler_metric=None,
                 step_scheduler_each_batch=True, verbose=False, args_l=(), kwargs_l=None, args_b=(), kwargs_b=None,
                 args_a=(), kwargs_a=None, grad_clip_norm=None, use_amp=False, grad_accum_steps=1
                ):

        self.configs = configs
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_on_validation = eval_on_validation
        self.test_loader = test_loader
        self.function_container = function_container
        self.loss_meter = AverageMeter()
        self.performance_meter = AverageMeter()
        self.scheduler = scheduler
        self.scheduler_metric = scheduler_metric
        self.step_scheduler_each_batch = step_scheduler_each_batch
        self.verbose = verbose
        self.args_l = args_l
        self.kwargs_l = kwargs_l or {}
        self.args_b = args_b
        self.kwargs_b = kwargs_b or {}
        self.args_a = args_a
        self.kwargs_a = kwargs_a or {}
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self._scaler = torch.amp.GradScaler(device='cuda',enabled=self.use_amp)

    def validate(self):
        if self.eval_on_validation:
            if self.val_loader is None:
                raise ValueError("Validation loader is required for evaluation.")


    def _one_epoch_train(self, epoch, run=None, logger_update_at_each_epoch=True):
        self.model.train()
        loader_len = len(self.train_loader)
        accumulated_loss = 0.
        n_steps = 0
        n_step_for_update = 0



        with tqdm(total=loader_len, desc=f'Epoch {epoch + 1}', file=sys.stdout, colour='green', ncols=100, dynamic_ncols=True) as pbar:
            self.optimizer.zero_grad(set_to_none=True)
            for idx, batch in enumerate(self.train_loader):
                batch, y = self.function_container.batch_extractor(batch, *self.args_b, **self.kwargs_b)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                        loss = self.function_container.loss_function(output, y, *self.args_l, **self.kwargs_l)
                        loss_to_backprop = loss / self.grad_accum_steps
                    self._scaler.scale(loss_to_backprop).backward()
                else:
                    output = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                    loss = self.function_container.loss_function(output, y, *self.args_l, **self.kwargs_l)
                    loss_to_backprop = loss / self.grad_accum_steps
                    loss_to_backprop.backward()

                accumulated_loss += loss.item()
                n_steps += 1
                n_step_for_update += 1

                do_step = ((idx + 1) % self.grad_accum_steps == 0) or (idx + 1 == loader_len)
                if do_step:
                    if self.grad_clip_norm is not None:
                        if self.use_amp:
                            self._scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    if self.use_amp:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    if run is not None and not logger_update_at_each_epoch and n_step_for_update % 50 == 0:
                        run.log({"loss": accumulated_loss / n_steps})

                    accumulated_loss = 0.
                    n_steps = 0

                self.loss_meter.update(N(loss))

                if self.scheduler and self.step_scheduler_each_batch and do_step:
                    SchedulerWrapper(self.scheduler).step(metric=self.scheduler_metric)

                pbar.update(1)
                pbar.set_postfix({'avg_loss': self.loss_meter.avg})

            if self.eval_on_validation:
                self.model.eval()
                with torch.no_grad():
                    score = self.function_container.validation_performance(
                        self.model,
                        self.val_loader,
                        *self.args_a,
                        **self.kwargs_a
                    )
                self.performance_meter.update(score['score'])
                pbar.set_postfix({
                    'avg_loss': self.loss_meter.avg,
                    # 'score': self.performance_meter.avg
                    **score
                    }
                )

            if run is not None:
                if logger_update_at_each_epoch:
                    # run.log({"score": self.performance_meter.avg, "loss": self.loss_meter.avg})
                    run.log({"loss": self.loss_meter.avg, **score})
                else:
                    # run.log({"score": self.performance_meter.avg})
                    run.log({**score})



    def train(self):
        early_stopping = EarlyStopping(self.configs, verbose=self.verbose) if self.configs.require_early_stop else None
        run = self.configs.logger.run if self.configs.logger is not None else None

        # train loop
        for epoch in range(self.configs.num_epochs):
            self._one_epoch_train(epoch, run, self.configs.logger_update_at_each_epoch)

            if self.scheduler and not self.step_scheduler_each_batch:
                SchedulerWrapper(self.scheduler).step(metric=self.scheduler_metric)

            if self.configs.require_early_stop:
                early_stopping(self.performance_meter.val, self.model)
                if early_stopping.early_stop:
                    print(f"Training stopped after epoch {epoch} by early stopping.")
                    break

            self.performance_meter.reset()
            self.loss_meter.reset()

        if run is not None:
            run.finish()


    def test(self):
        if self.test_loader is None:
            raise ValueError("Test loader is required.")

        loader_len = len(self.test_loader)
        self.model.eval()

        with torch.no_grad():
            with tqdm(total=loader_len, desc='TEST', file=sys.stdout, colour='blue', ncols=100, dynamic_ncols=True) as pbar:
                score = self.function_container.test_performance(
                    model=self.model,
                    loader=self.test_loader,
                    pbar=pbar
                )
                pbar.set_postfix({'score': score})