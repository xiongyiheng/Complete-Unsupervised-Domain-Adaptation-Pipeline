import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


class BNM(nn.Module):

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
    ):
        super(BNM, self).__init__()

        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.transfer_loss_weight = transfer_loss_weight

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
        p_s = self.classifier(self.backbone(x_s))
        softmax_t = F.softmax(self.classifier(self.backbone(x_t)), dim=1)

        cls_loss = F.cross_entropy(p_s, y_s, weight=ce_weight)

        _, s_t, _ = torch.svd(softmax_t)
        transfer_loss = -torch.mean(s_t)

        total_loss = cls_loss + self.transfer_loss_weight * transfer_loss

        return total_loss, cls_loss, transfer_loss
