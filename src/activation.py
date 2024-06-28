import math
from typing import Optional

import torch

# Importable container for available activations
ACTIVATIONS = {
    "relu": torch.nn.ReLU
}