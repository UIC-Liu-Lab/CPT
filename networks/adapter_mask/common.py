from dataclasses import dataclass
from typing import Union, Callable

import torch.nn as nn


@dataclass
class AdapterMaskConfig:
    hidden_size: int
    adapter_size: int
    ffn_adapter_size: int
    attn_adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float
    ntasks: int
    smax: int
    mode: str = "sequential"  # "sequential" / "parallel"

    def __post_init__(self):
        if self.mode not in ("sequential", "parallel"):
            raise NotImplementedError(f"The current mode {self.mode} is not supported!")


def freeze_all_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model
