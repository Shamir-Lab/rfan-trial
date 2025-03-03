from pathlib import Path
import torch

from src import models


def directory_deep_kernel_gp(base_dir, config):
    # Get model parameters from config
    kernel = config.get("kernel")
    num_inducing_points = config.get("num_inducing_points")
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    return (
        Path(base_dir)
        / "dk_gp"
        / f"k-{kernel}_ip-{num_inducing_points}-dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
    )


def is_model_trained(model):
    for param in model.parameters():
        if param.requires_grad and param.grad is not None and torch.norm(param.grad) != 0:
            return True
    return False


def train_deep_kernel_gp(ds_train, ds_valid, job_dir, config, dim_input, overwrite_model=False):
    """ Initializes and train model """
    if (not (job_dir / "best_checkpoint.pt").exists()) or overwrite_model:
        # Get model parameters from config
        kernel = config.get("kernel")
        num_inducing_points = config.get("num_inducing_points")
        dim_hidden = config.get("dim_hidden")
        dim_output = config.get("dim_output")
        depth = config.get("depth")
        negative_slope = config.get("negative_slope")
        dropout_rate = config.get("dropout_rate")
        spectral_norm = config.get("spectral_norm")
        learning_rate = config.get("learning_rate")
        batch_size = config.get("batch_size")
        epochs = config.get("epochs")
        train_ratio = config.get("train_ratio")
        model = models.DeepKernelGP(
            job_dir=job_dir,
            kernel=kernel,
            num_inducing_points=num_inducing_points,
            inducing_point_dataset=ds_train,
            architecture="resnet",
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_train),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            train_ratio=train_ratio,
            patience=5,
            num_workers=0,
            seed=config.get("seed"),
        )
        _ = model.fit(ds_train, ds_valid)


def predict_deep_kernel_gp(dataset, job_dir, config):
    # Get model parameters from config
    kernel = config.get("kernel")
    num_inducing_points = config.get("num_inducing_points")
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    train_ratio = config.get("train_ratio")
    model = models.DeepKernelGP(
        job_dir=job_dir,
        kernel=kernel,
        num_inducing_points=num_inducing_points,
        inducing_point_dataset=dataset,
        architecture="resnet",
        dim_input=dataset.dim_input,
        dim_hidden=dim_hidden,
        dim_output=dim_output,
        depth=depth,
        negative_slope=negative_slope,
        batch_norm=False,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(dataset),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        train_ratio=train_ratio,
        patience=5,
        num_workers=0,
        seed=config.get("seed"),
    )
    model.load()
    return model.predict_mus(dataset)


DIRECTORIES = {"deep_kernel_gp": directory_deep_kernel_gp}
TRAIN_FUNCTIONS = {"deep_kernel_gp": train_deep_kernel_gp}
PREDICT_FUNCTIONS = {"deep_kernel_gp": predict_deep_kernel_gp}