import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Tuple
import numpy as np

from .classifier import classifier

class SourceOnly(nn.Module):
    def __init__(self, backbone, in_features, num_classes, is_nonlinear, dropout):
        super(SourceOnly, self).__init__()
        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
 
    def forward(self, x_s, x_t):
        f_s = self.backbone(x_s)
        f_t = self.backgone(x_t)
        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)
        return p_s, p_t
 
    def compute_loss(self, x_s, y_s, x_t, ce_weight):
        f_s = self.feature_extractor(x_s)
        p_s = self.classifier(f_s)
        cls_loss = F.cross_entropy(p_s, y_s, ce_weight)
 
        total_loss = cls_loss
        return total_loss
