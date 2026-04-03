from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class ModelConfig:
    # Input
    in_channels: int
    n_cyclical_features: int

    # Conv blocks
    conv_filters: List[int]
    kernel_size: int
    stride: int
    padding: int
    use_batch_norm: bool
    activation: str

    # Pooling
    pooling: str

    # Dense head
    dense_hidden: List[int]
    dropout: float

    # Output mode
    output_capacity_factor: bool  # If True, sigmoid output in [0,1]

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        m = raw["model"]
        return cls(
            in_channels=m["in_channels"],
            n_cyclical_features=m["n_cyclical_features"],
            conv_filters=m["conv_filters"],
            kernel_size=m["kernel_size"],
            stride=m["stride"],
            padding=m["padding"],
            use_batch_norm=m["use_batch_norm"],
            activation=m["activation"],
            pooling=m["pooling"],
            dense_hidden=m["dense_hidden"],
            dropout=m["dropout"],
            output_capacity_factor=m.get("output_capacity_factor", False),
        )


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    max_epochs: int
    optimizer: str
    weight_decay: float  # L2 regularisation — 0.0 disables it

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        t = raw["training"]
        return cls(
            learning_rate=t["learning_rate"],
            batch_size=t["batch_size"],
            max_epochs=t["max_epochs"],
            optimizer=t["optimizer"],
            weight_decay=t.get("weight_decay", 0.0),
        )
