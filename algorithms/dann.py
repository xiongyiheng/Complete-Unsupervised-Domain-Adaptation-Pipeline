import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple
import numpy as np

from .classifier import classifier 

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000, auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1

class DANN(nn.Module):
    def __init__(self, backbone, in_features, num_classes, is_nonlinear, dropout, transfer_loss_weight, max_iters):
        super(DANN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.discriminator = classifier(in_features, 2, is_nonlinear, dropout)
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                                 max_iters=max_iters,
                                                 auto_step=True)
        self.transfer_loss_weight = transfer_loss_weight

    def forward(self, x_s, x_t):
        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)
        
        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)
        
        return p_s, p_t

    def compute_loss(self, x_s, y_s, x_t, ce_weight):
        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)
        
        p_s = self.classifier(f_s)
        cls_loss = F.cross_entropy(p_s, y_s)

        f = torch.cat([f_s, f_t], dim=0)
        reversed_f = self.grl(f)
        disc_out = self.discriminator(reversed_f)
        
        domain_labels = torch.cat([
            torch.zeros(f_s.size(0), dtype=torch.long),
            torch.ones(f_t.size(0), dtype=torch.long)
        ], dim=0).to(f_s.device)
        
        disc_loss = F.cross_entropy(disc_out, domain_labels, ce_weight)
        
        total_loss = cls_loss + (self.transfer_loss_weight * disc_loss)
        
        return total_loss, cls_loss, disc_loss
