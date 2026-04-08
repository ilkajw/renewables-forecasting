import torch
import torch.nn as nn
from typing import List

from renewables_forecasting.models.config import ModelConfig


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def _build_conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_batch_norm: bool,
        activation: str,
) -> nn.Sequential:
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


def _build_dense_head(
        in_features: int,
        hidden_sizes: List[int],
        dropout: float,
        activation: str,
        output_sigmoid: bool = False
) -> nn.Sequential:
    """
    Builds the dense head that takes the concatenated CNN output and cyclical
    time features and maps them to a scalar generation estimate.

    The final layer is always a single linear unit with no activation,
    giving an unbounded scalar output appropriate for regression.
    """
    layers = []
    current_in = in_features
    for hidden in hidden_sizes:
        layers.extend([
            nn.Linear(current_in, hidden),
            _ACTIVATIONS[activation](),
            nn.Dropout(dropout),
        ])
        current_in = hidden
    layers.append(nn.Linear(current_in, 1))
    if output_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class GenerationCNN(nn.Module):
    """
    CNN for predicting hourly renewable energy generation from stacked
    spatial grids of weather variables and installed capacity.

    Architecture:
        1. A stack of conv blocks processes the (C × lat × lon) input grid,
           learning spatially-aware weather-capacity interactions.
        2. Global pooling collapses spatial dimensions to a fixed-size vector,
           making the model invariant to grid size.
        3. Cyclical time features (doy_sin, doy_cos, hod_sin, hod_cos) are
           concatenated to the pooled spatial encoding.
        4. A dense head maps the combined vector to a scalar generation estimate.

    The depth and width of both the conv stack and dense head are fully
    controlled by the ModelConfig, allowing architecture search without
    changing any model code.

    Parameters
    ----------
    config:
        ModelConfig instance loaded from a YAML config file.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{config.activation}'. "
                f"Choose from: {list(_ACTIVATIONS.keys())}"
            )
        if config.pooling not in ("global_average", "global_max"):
            raise ValueError(
                f"Unknown pooling '{config.pooling}'. "
                f"Choose from: global_average, global_max"
            )

        # ── Conv blocks ───────────────────────────────────────────────────────
        conv_blocks = []
        in_ch = config.in_channels
        for out_ch in config.conv_filters:
            conv_blocks.append(
                _build_conv_block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    use_batch_norm=config.use_batch_norm,
                    activation=config.activation,
                )
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)

        # ── Spatial pooling ───────────────────────────────────────────────────
        if config.pooling == "global_average":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

        # ── Dense head ────────────────────────────────────────────────────────
        # Input size = last conv layer's filters + cyclical time features
        dense_in = config.conv_filters[-1] + config.n_cyclical_features
        self.dense = _build_dense_head(
            in_features=dense_in,
            hidden_sizes=config.dense_hidden,
            dropout=config.dropout,
            activation=config.activation,
            output_sigmoid=config.output_capacity_factor
        )

        self.config = config

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Spatial input grid of shape (batch, in_channels, lat, lon).
            Channels are ordered as defined in the config, e.g.
            [ssrd, t2m, capacity_kw] for solar.
        time_features:
            Cyclical time encodings of shape (batch, n_cyclical_features),
            i.e. [doy_sin, doy_cos, hod_sin, hod_cos].

        Returns
        -------
        torch.Tensor of shape (batch,) — scalar generation estimate per sample.
        """
        # Spatial encoding: (batch, C, lat, lon) → (batch, conv_filters[-1])
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)  # Remove spatial dims after pooling

        # Concatenate time features: (batch, conv_filters[-1] + n_cyclical)
        x = torch.cat([x, time_features], dim=1)

        # Dense head: (batch, ...) → (batch, 1) → (batch,)
        return self.dense(x).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    