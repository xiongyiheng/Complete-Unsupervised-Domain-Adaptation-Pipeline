import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Tuple
import numpy as np

def classifier(in_features, out_features, is_nonlinear=False, dropout=0.0):
    if is_nonlinear:
        layers = nn.Sequential(
            nn.Linear(in_features, in_features // 2, bias=False),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(in_features // 2, in_features // 4, bias=False),
            nn.BatchNorm1d(in_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(in_features // 4, out_features)
        )

        for m in layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return layers
    else:
        return torch.nn.Linear(in_features, out_features)

