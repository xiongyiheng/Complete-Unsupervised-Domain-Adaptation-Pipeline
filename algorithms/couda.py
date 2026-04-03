import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


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


class CoUDA(nn.Module):

    def __init__(
        self,
        backbone1,
        backbone2,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
        grl_max_iters: int = 2500,
    ):
        super(CoUDA, self).__init__()

        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.cls_head1 = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.cls_head2 = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.discriminator = classifier(in_features, 2, is_nonlinear, dropout)
        self.grl1 = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=grl_max_iters, auto_step=True)
        self.grl2 = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=grl_max_iters, auto_step=True)
        self.tlw = transfer_loss_weight

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        p_s = 0.5 * (
            self.cls_head1(self.backbone1(x_s)) +
            self.cls_head2(self.backbone2(x_s))
        )
        p_t = 0.5 * (
            self.cls_head1(self.backbone1(x_t)) +
            self.cls_head2(self.backbone2(x_t))
        )
        return p_s, p_t

    def compute_loss(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
        ce_weight: torch.Tensor,
    ):
        x_all = torch.cat([x_s, x_t], dim=0)
        domain_labels = torch.cat([
            torch.zeros(x_s.size(0), dtype=torch.long, device=x_s.device),
            torch.ones(x_t.size(0), dtype=torch.long, device=x_s.device),
        ])

        f1 = self.backbone1(x_all)
        y1 = self.cls_head1(f1)
        disc_loss1 = F.cross_entropy(self.discriminator(self.grl1(f1)), domain_labels, reduction='none')

        f2 = self.backbone2(x_all)
        y2 = self.cls_head2(f2)
        disc_loss2 = F.cross_entropy(self.discriminator(self.grl2(f2)), domain_labels, reduction='none')

        p1 = F.softmax(y1, dim=1)
        p2 = F.softmax(y2, dim=1)

        cos_sim = F.cosine_similarity(p1, p2, dim=1, eps=1e-8)
        lam = 2.0 - cos_sim
        disc_loss = ((disc_loss1 + disc_loss2) * lam).mean()

        cls_loss1 = F.cross_entropy(y1[:x_s.size(0)], y_s, weight=ce_weight)
        cls_loss2 = F.cross_entropy(y2[:x_s.size(0)], y_s, weight=ce_weight)
        cls_loss = cls_loss1 + cls_loss2

        eps = 1e-8
        p1 = p1.clamp(min=eps)
        p2 = p2.clamp(min=eps)
        y_bar = (0.5 * (p1 + p2)).clamp(min=eps)
        kl1 = (p1 * (p1.log() - y_bar.log())).sum(dim=1)
        kl2 = (p2 * (p2.log() - y_bar.log())).sum(dim=1)
        dev_loss = (kl1 + kl2).mean()

        transfer_loss = disc_loss + self.tlw * dev_loss
        total_loss = cls_loss + transfer_loss

        return total_loss, cls_loss, transfer_loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.cls_head1(self.backbone1(x)) +
            self.cls_head2(self.backbone2(x))
        )

