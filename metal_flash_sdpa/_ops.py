import torch

# Import the extension module to ensure Torch operator registration happens.
from . import _C  # noqa: F401

# The C++ extension registers ops under this namespace:
#   torch.ops.metal_flash_sdpa.flash_attention_varlen(...)
ops = torch.ops.metal_flash_sdpa

__all__ = ["ops"]

