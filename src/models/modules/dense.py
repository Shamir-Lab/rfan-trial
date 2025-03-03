# Defines custom neural network with configurable architecture, including activation, dropout, normalization, etc.
# Originated from https://github.com/OATML/causal-bald/tree/fdb69553837edde0f97a5a8a647a6a24a51077a2

from torch import nn

from src.models.modules.spectral_norm import spectral_norm_fc


class Activation(nn.Module):
    """
    Implements an activation module with optional batch normalization and dropout.

    Args:
        dim_input (int): Number of input features.
        negative_slope (float): Slope of the LeakyReLU activation function (if applicable).
        dropout_rate (float): Dropout probability.
        batch_norm (bool): Whether to apply batch normalization.
    """
    def __init__(
        self,
        dim_input,
        negative_slope,
        dropout_rate,
        batch_norm,
    ):
        super(Activation, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm1d(num_features=dim_input) if batch_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=negative_slope)
            if negative_slope >= 0.0
            else nn.ELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs):
        return self.op(inputs)


class PreactivationDense(nn.Module):
    """
    Applies activation (LeakyReLU or ELU), dropout, linear transformation (FC layer), with optional spectral norm.

    Args:
        dim_input (int): Number of input features.
        dim_output (int): Number of output features.
        bias (bool): Whether to include bias in the linear transformation.
        negative_slope (float): Slope of the LeakyReLU activation function.
        dropout_rate (float): Dropout probability.
        batch_norm (bool): Whether to apply batch normalization.
        spectral_norm (float): Strength of spectral normalization (if applicable).
    """
    def __init__(
        self,
        dim_input,
        dim_output,
        bias,
        negative_slope,
        dropout_rate,
        batch_norm,
        spectral_norm,
    ):
        super(PreactivationDense, self).__init__()
        self.op = nn.Sequential(
            Activation(
                dim_input=dim_input,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
            )
        )
        linear = nn.Linear(in_features=dim_input, out_features=dim_output, bias=bias)
        self.op.add_module(
            "linear",
            spectral_norm_fc(linear, spectral_norm) if spectral_norm > 0.0 else linear,
        )

    def forward(self, inputs):
        return self.op(inputs)


class ResidualDense(nn.Module):
    """
    Implements a residual connection for dense layers, defined as PreactivationDense(x)+x.

    Args:
        dim_input (int): Number of input features.
        dim_output (int): Number of output features.
        bias (bool): Whether to include bias in the linear transformations.
        negative_slope (float): Slope of the LeakyReLU activation function.
        dropout_rate (float): Dropout probability.
        batch_norm (bool): Whether to apply batch normalization.
        spectral_norm (float): Strength of spectral normalization (if applicable).
    """
    def __init__(
        self,
        dim_input,
        dim_output,
        bias,
        negative_slope,
        dropout_rate,
        batch_norm,
        spectral_norm,
    ):
        super(ResidualDense, self).__init__()
        if dim_input != dim_output:
            self.shortcut = nn.Sequential(nn.Dropout(p=dropout_rate))
            linear = nn.Linear(
                in_features=dim_input, out_features=dim_output, bias=bias
            )
            self.shortcut.add_module(
                "linear",
                spectral_norm_fc(linear, spectral_norm)
                if spectral_norm > 0.0
                else linear,
            )
        else:
            self.shortcut = nn.Identity()

        self.op = PreactivationDense(
            dim_input=dim_input,
            dim_output=dim_output,
            bias=bias,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
        )

    def forward(self, inputs):
        return self.op(inputs) + self.shortcut(inputs)


MODULES = {"basic": PreactivationDense, "resnet": ResidualDense}


class NeuralNetwork(nn.Module):
    """
    Implements a neural network with configurable architecture.

    Args:
        architecture (str): Architecture type ("basic" for dense, "resnet" for residual).
        dim_input (int): Number of input features.
        dim_hidden (int): Number of hidden features.
        depth (int): Number of hidden layers.
        negative_slope (float): Slope of the LeakyReLU activation function.
        batch_norm (bool): Whether to apply batch normalization.
        dropout_rate (float): Dropout probability.
        spectral_norm (float): Strength of spectral normalization (if applicable).
        activate_output (bool): Whether to apply an output activation function.
    """
    def __init__(
        self,
        architecture,
        dim_input,
        dim_hidden,
        depth,
        negative_slope,
        batch_norm,
        dropout_rate,
        spectral_norm,
        activate_output=True,
    ):
        super(NeuralNetwork, self).__init__()
        self.op = nn.Sequential()
        hidden_module = MODULES[architecture]
        for i in range(depth):
            if i == 0:
                input_layer = nn.Linear(
                    in_features=dim_input,
                    out_features=dim_hidden,
                    bias=not batch_norm,
                )
                self.op.add_module(
                    name="input_layer",
                    module=spectral_norm_fc(input_layer, spectral_norm)
                    if spectral_norm > 0.0
                    else input_layer,
                )
            else:
                self.op.add_module(
                    name="hidden_layer_{}".format(i),
                    module=hidden_module(
                        dim_input=dim_hidden,
                        dim_output=dim_hidden,
                        bias=not batch_norm,
                        negative_slope=negative_slope,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm,
                        spectral_norm=spectral_norm,
                    ),
                )
        if activate_output:
            self.op.add_module(
                name="output_activation",
                module=Activation(
                    dim_input=dim_hidden,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                ),
            )
        self.dim_output = dim_hidden

    def forward(self, inputs):
        return self.op(inputs)