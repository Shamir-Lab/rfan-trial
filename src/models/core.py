# Provides base classes and utilities for building and training PyTorch models
# Originated from https://github.com/OATML/causal-bald/tree/fdb69553837edde0f97a5a8a647a6a24a51077a2

import os
import copy
import torch

from abc import ABC

from ray import tune

from ignite import utils
from ignite import engine
from ignite import distributed

from torch.utils import data
from torch.utils import tensorboard

from src import datasets


class BaseModel(ABC):
    """
    Abstract base class for general models.

    Attributes:
        job_dir (str): The directory path to save model-related artifacts.
        seed (int): The random seed for reproducibility.

    Methods:
        fit(train_dataset, tune_dataset): Train the model on the given datasets.
        save(): Save the model's state to the specified directory.
        load(): Load the model's state from a saved checkpoint.
    """
    def __init__(self, job_dir, seed):
        super(BaseModel, self).__init__()
        self.job_dir = job_dir
        self.seed = seed

    def fit(self, train_dataset, tune_dataset):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement train()"
        )

    def save(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement save()"
        )

    def load(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement load()"
        )


class PyTorchModel(BaseModel):
    """
    Base class for PyTorch-based models.

    Attributes:
        learning_rate (float): The learning rate for optimization.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        num_workers (int): Number of workers for data loading.

    Methods:
        train_step(engine, batch): Perform a training step on a batch of data.
        tune_step(engine, batch): Perform a tuning/validation step on a batch of data.
        preprocess(batch): Preprocess a batch of data for device placement.
        on_epoch_completed(engine, train_loader, tune_loader): Callback executed at the end of each epoch.
        on_training_completed(engine, loader): Callback executed when training is completed.
    """
    def __init__(self, job_dir, learning_rate, batch_size, epochs, num_workers, seed):
        super(PyTorchModel, self).__init__(job_dir=job_dir, seed=seed)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.summary_writer = tensorboard.SummaryWriter(log_dir=self.job_dir)
        self.logger = utils.setup_logger(
            name=__name__ + "." + self.__class__.__name__, distributed_rank=0
        )
        self.trainer = engine.Engine(self.train_step)
        self.evaluator = engine.Engine(self.tune_step)
        self._network = None
        self._optimizer = None
        self._metrics = None
        self.likelihood = None
        self.num_workers = num_workers
        self.device = distributed.device()
        self.best_state = None
        self.counter = 0

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def train_step(self, trainer, batch):
        raise NotImplementedError()

    def tune_step(self, trainer, batch):
        raise NotImplementedError()

    def preprocess(self, batch):
        inputs, targets = batch
        inputs = (
            [x.to(self.device) for x in inputs]
            if isinstance(inputs, list)
            else inputs.to(self.device)
        )
        targets = (
            [x.to(self.device) for x in targets]
            if isinstance(targets, list)
            else targets.to(self.device)
        )
        return inputs, targets

    def fit(self, train_dataset, tune_dataset):
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=datasets.RandomFixedLengthSampler(
                train_dataset, 100 * self.batch_size
            ),
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        tune_loader = data.DataLoader(
            tune_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        )
        # Instantiate trainer
        for k, v in self.metrics.items():
            v.attach(self.trainer, k)
            v.attach(self.evaluator, k)
        self.trainer.add_event_handler(
            engine.Events.EPOCH_COMPLETED,
            self.on_epoch_completed,
            train_loader,
            tune_loader,
        )
        self.trainer.add_event_handler(
            engine.Events.COMPLETED, self.on_training_completed, tune_loader
        )
        # Train
        self.trainer.run(train_loader, max_epochs=self.epochs)
        return self.evaluator.state.metrics

    def on_epoch_completed(self, engine, train_loader, tune_loader):
        train_metrics = self.trainer.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in train_metrics) + 2
        for k, v in train_metrics.items():
            if type(v) == float:
                print("train {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("train {:<{justify}} {:<5}".format(k, v, justify=justify))
        self.evaluator.run(tune_loader)
        tune_metrics = self.evaluator.state.metrics
        if tune.is_session_enabled():
            tune.report(mean_loss=tune_metrics["loss"])
        justify = max(len(k) for k in tune_metrics) + 2
        for k, v in tune_metrics.items():
            if type(v) == float:
                print("tune {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
        if tune_metrics["loss"] < self.best_loss:
            self.best_loss = tune_metrics["loss"]
            self.counter = 0
            self.update()
        else:
            self.counter += 1
        if self.counter == self.patience:
            self.logger.info(
                "Early Stopping: No improvement for {} epochs".format(self.patience)
            )
            engine.terminate()

    def on_training_completed(self, engine, loader):
        self.save()
        self.load()
        if not tune.is_session_enabled():
            self.evaluator.run(loader)
            metric_values = self.evaluator.state.metrics
            print("Metrics Epoch", engine.state.epoch)
            justify = max(len(k) for k in metric_values) + 2
            for k, v in metric_values.items():
                if type(v) == float:
                    print("best {:<{justify}} {:<5f}".format(k, v, justify=justify))
                    continue

    def update(self):
        if not tune.is_session_enabled():
            self.best_state = {
                "model": copy.deepcopy(self.network.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "engine": copy.copy(self.trainer.state),
            }
            if self.likelihood is not None:
                self.best_state["likelihood"] = copy.deepcopy(
                    self.likelihood.state_dict()
                )

    def save(self):
        if not tune.is_session_enabled():
            p = os.path.join(self.job_dir, "best_checkpoint.pt")
            torch.save(self.best_state, p)

    def load(self):
        if tune.is_session_enabled():
            with tune.checkpoint_dir(step=self.trainer.state.epoch) as checkpoint_dir:
                p = os.path.join(checkpoint_dir, "checkpoint.pt")
        else:
            file_name = "best_checkpoint.pt"
            p = os.path.join(self.job_dir, file_name)
        if not os.path.exists(p):
            self.logger.info(
                "Checkpoint {} does not exist, starting a new engine".format(p)
            )
            return
        self.logger.info("Loading saved checkpoint {}".format(p))
        checkpoint = torch.load(p)
        self.network.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.likelihood is not None:
            self.likelihood.load_state_dict(checkpoint["likelihood"])
        self.trainer.state = checkpoint["engine"]
