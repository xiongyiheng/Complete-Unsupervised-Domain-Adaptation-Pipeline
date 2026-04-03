import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier import classifier


class GradientReverseFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: float = 1.) -> torch.Tensor:
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
        max_iters: int = 1000,
        auto_step: bool = False,
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


class AD2A(nn.Module):

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: list,
        warmup_steps: int = 3000,
        grl_max_iters: int = 1425,
    ):
        super(AD2A, self).__init__()

        self.backbone = backbone
        self.cls_head = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.discriminator = classifier(in_features, 2, is_nonlinear, dropout)
        self.grl = WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=grl_max_iters, auto_step=True
        )

        self.register_buffer('update_count', torch.tensor(0))
        self.warmup_steps = warmup_steps
        self.tlw = _TLW_PRESETS.get(transfer_loss_weight, _TLW_DEFAULT)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        p_s = self.cls_head(self.backbone(x_s))
        p_t = self.cls_head(self.backbone(x_t))
        return p_s, p_t

    def compute_loss(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
        ce_weight: torch.Tensor,
    ):
        self.update_count += 1

        x_all = torch.cat([x_s, x_t], dim=0)
        f_all = self.backbone(x_all)
        y_all = self.cls_head(f_all)

        y_s_out = y_all[:x_s.size(0)]
        cls_loss = F.cross_entropy(y_s_out, y_s, weight=ce_weight)

        transfer_loss = torch.tensor(0.0, device=x_s.device)

        if self.update_count.item() >= self.warmup_steps:
            reversed_f = self.grl(f_all)
            disc_out = self.discriminator(reversed_f)

            domain_labels = torch.cat([
                torch.zeros(x_s.size(0), dtype=torch.long, device=x_s.device),
                torch.ones(x_t.size(0), dtype=torch.long, device=x_s.device)
            ])
            disc_loss = F.cross_entropy(disc_out, domain_labels)

            attention = self.backbone.spatial_gate.attention.squeeze()
            att_s = attention[:x_s.size(0)]
            att_t = attention[x_s.size(0):]
            att_loss = ((att_s - att_t) ** 2).mean()

            transfer_loss = self.tlw[0] * disc_loss + self.tlw[1] * att_loss

        total_loss = cls_loss + transfer_loss

        return total_loss, cls_loss, transfer_loss

