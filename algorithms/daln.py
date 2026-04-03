import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


class GradientReverseFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        return input * 1.0

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        max_iters: int = 1500,
        auto_step: bool = True
    ):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + math.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.iter_num += 1
        return GradientReverseFunction.apply(input, coeff)


class DALN(nn.Module):

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
        grl_max_iters: int = 1500,
    ):
        super(DALN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.transfer_loss_weight = transfer_loss_weight
        
        self.grl = WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=grl_max_iters, auto_step=True
        )

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        p_s = self.classifier(self.backbone(x_s))
        p_t = self.classifier(self.backbone(x_t))
        return p_s, p_t

    def compute_loss(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
        ce_weight: torch.Tensor,
    ):
        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)

        y_s_out = self.classifier(f_s)
        cls_loss = F.cross_entropy(y_s_out, y_s, weight=ce_weight)

        f_all = torch.cat([f_s, f_t], dim=0)
        f_grl = self.grl(f_all)
        y_all_grl = self.classifier(f_grl)

        y_s_grl = y_all_grl[:f_s.size(0)]
        y_t_grl = y_all_grl[f_s.size(0):]

        pre_s = F.softmax(y_s_grl, dim=1)
        pre_t = F.softmax(y_t_grl, dim=1)

        discrepancy = (-torch.norm(pre_t, p='nuc') + torch.norm(pre_s, p='nuc')) / pre_t.size(0)
        transfer_loss = -discrepancy

        total_loss = cls_loss + self.transfer_loss_weight * transfer_loss

        return total_loss, cls_loss, transfer_loss
