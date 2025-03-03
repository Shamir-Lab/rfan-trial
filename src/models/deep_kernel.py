"""
Defines a Deep Kernel Gaussian Process model: The model uses a deep feature extractor (NN encoder) to transform the
inputs and defines a Gaussian process (GP) kernel over the extracted feature representation to make predictions.

Code is originated from https://github.com/OATML/causal-bald/tree/fdb69553837edde0f97a5a8a647a6a24a51077a2
"""

import torch

from torch import nn
from torch import optim
from torch.utils import data

from gpytorch import mlls
from gpytorch import likelihoods

from ignite import metrics

from src.utils import set_seeds
from src.models import core
from src.models.modules import dense
from src.models.modules import gaussian_process


class DeepKernelGP(core.PyTorchModel):
    """
    Deep Kernel GP model.
    The model combines neural network encoder (feature representation) with Gaussian process predictions.

    Args:
        job_dir (str): Directory to save model checkpoints and logs.
        kernel (str): Type of kernel for the Gaussian process.
        num_inducing_points (int): Number of inducing points.
        inducing_point_dataset (torch.utils.data.Dataset): Dataset for inducing points.
        architecture (str): Neural network architecture configuration ("resnet" or "basic')
        dim_input, dim_hidden, dim_output (int): Dimensionality of input features, hidden layers, and output.
        depth (int): Depth of the neural network.
        negative_slope (float): Negative slope for LeakyReLU.
        batch_norm (bool): Whether to use batch normalization.
        spectral_norm (bool): Whether to use spectral normalization.
        dropout_rate (float): Dropout rate.
        weight_decay (float): Weight decay for optimization.
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        num_workers (int): Number of workers for data loading.
        seed (int): Random seed for reproducibility.
    """
    def __init__(
        self,
        job_dir,
        kernel,
        num_inducing_points,
        inducing_point_dataset,
        architecture,
        dim_input,
        dim_hidden,
        dim_output,
        depth,
        negative_slope,
        batch_norm,
        spectral_norm,
        dropout_rate,
        weight_decay,
        learning_rate,
        batch_size,
        epochs,
        train_ratio,
        patience,
        num_workers,
        seed,
    ):
        super(DeepKernelGP, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        # Deep feature encoder (in DUE the encoder contains residual connections and spectral normalization.
        self.encoder = nn.Sequential(
            dense.NeuralNetwork(
                architecture=architecture,
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                depth=depth,
                negative_slope=negative_slope,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                spectral_norm=spectral_norm,
                activate_output=False,
            ),
            dense.Activation(
                dim_input=None,
                negative_slope=negative_slope,
                dropout_rate=0.0,
                batch_norm=batch_norm,
            ),
        )
        self.encoder.to(self.device)

        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        (initial_inducing_points, initial_lengthscale) = gaussian_process.initial_values_for_GP(
                                                                    train_dataset=inducing_point_dataset,
                                                                    feature_extractor=self.encoder,
                                                                    n_inducing_points=num_inducing_points,
                                                                    device=self.device,
                                                                    train_ratio=train_ratio)

        # Define GP
        self.gp = gaussian_process.VariationalGP(
            num_outputs=dim_output,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            separate_inducing_points=False,
            kernel=kernel,
            ard=None,
            lengthscale_prior=False,
        ).to(self.device)

        # Define DeepKernelGP: encoder and then GP
        self.network = gaussian_process.DeepKernelGP(
            encoder=self.encoder,
            gp=self.gp)

        self.likelihood = likelihoods.GaussianLikelihood()
        self.optimizer = optim.Adam(
            params=[
                {"params": self.encoder.parameters(), "lr": self.learning_rate},
                {"params": self.gp.parameters(), "lr": 2 * self.learning_rate},
                {"params": self.likelihood.parameters(), "lr": 2 * self.learning_rate},
            ],
            weight_decay=weight_decay,
        )
        self.loss = mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.network.gp,
            num_data=len(inducing_point_dataset),
        )
        self.metrics = {
            "loss": metrics.Average(
                output_transform=lambda x: -self.likelihood.expected_log_prob(
                    x["targets"].squeeze(), x["outputs"]
                ).mean(),
                device=self.device,
            )
        }
        self.network.to(self.device)
        self.likelihood.to(self.device)

    def train_step(self, engine, batch):
        self.network.train()
        self.likelihood.train()
        inputs, targets = self.preprocess(batch)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = -self.loss(outputs, targets.squeeze()).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step(self, engine, batch):
        self.network.eval()
        self.likelihood.eval()
        inputs, targets = self.preprocess(batch)
        with torch.no_grad():
            outputs = self.network(inputs)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def predict_mus(self, ds, batch_size=None):
        """
        Predicts the posterior mean responses for both treatment arms (t=0 and t=1) using the trained model.

        Args:
            ds (torch.utils.data.Dataset): The dataset for which predictions are to be made (usually the entire dataset)
            batch_size (int, optional): Batch size for data loading. If None, a default batch size is used.

        Returns:
            tuple: A tuple containing two numpy arrays of the predicted posterior responses for treatment t=0
            (first array) and for treatment t=1 (second array). Each of shape (num_samples, num_data) where num_samples
            is the number of posterior samples drawn and num_data is the number of data points in the dataset.
        """
        set_seeds(self.seed)

        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mu_0 = []
        mu_1 = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                covariates = torch.cat([batch[0][:, :-1], batch[0][:, :-1]], 0)
                treatments = torch.cat(
                    [
                        torch.zeros_like(batch[0][:, -1:]),
                        torch.ones_like(batch[0][:, -1:]),
                    ],
                    0,
                )
                inputs = torch.cat([covariates, treatments], -1)
                posterior_predictive = self.network(inputs)
                samples = posterior_predictive.sample(torch.Size([1000])) # Draw a sample (n=1000) from the posterior distribution
                mus = samples.chunk(2, dim=1)
                mu_0.append(mus[0])
                mu_1.append(mus[1])
        return (
            torch.cat(mu_0, 1).to("cpu").numpy(),
            torch.cat(mu_1, 1).to("cpu").numpy(),
        )
