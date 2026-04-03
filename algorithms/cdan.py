import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Tuple
import numpy as np

from .classifier import classifier


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: float) -> torch.Tensor:
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
    def __init__(self, alpha: float, lo: float, hi: float, max_iters: int, auto_step: bool):
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


class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, features_dim: int, num_classes: int, output_dim: int):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


def entropy(predictions: torch.Tensor, reduction: str) -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class CDAN(nn.Module):
    def __init__(self, backbone, in_features, num_classes, is_nonlinear, dropout,
                 transfer_loss_weight, max_iters, randomized, entropy_conditioning):
        super(CDAN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)

        if randomized:
            disc_input_dim = 1024
            self.map = RandomizedMultiLinearMap(in_features, num_classes, disc_input_dim)
        else:
            disc_input_dim = in_features * num_classes
            self.map = MultiLinearMap()

        self.discriminator = classifier(disc_input_dim, 1, is_nonlinear, dropout)
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                                 max_iters=max_iters,
                                                 auto_step=True)
        self.entropy_conditioning = entropy_conditioning
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

        f = torch.cat([f_s, f_t], dim=0)
        logits = self.classifier(f)

        p_s = logits[:f_s.size(0)]
        cls_loss = F.cross_entropy(p_s, y_s, weight=ce_weight)

        g = F.softmax(logits, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = torch.sigmoid(self.discriminator(h))

        batch_size = f.size(0)
        d_label = (torch.arange(batch_size, device=d.device).unsqueeze(1) % 2).float()

        if self.entropy_conditioning:
            weight = 1.0 + torch.exp(-entropy(g, reduction='none'))
            weight = weight / torch.sum(weight) * batch_size
            disc_loss = F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction='mean')
        else:
            disc_loss = F.binary_cross_entropy(d, d_label, reduction='mean')

        total_loss = cls_loss + (self.transfer_loss_weight * disc_loss)

        return total_loss, cls_loss, disc_loss
