import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


class MCD(nn.Module):

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
        lr: float,
        weight_decay: float,
        num_mcd_steps: int = 4,
    ):
        super(MCD, self).__init__()

        self.backbone = backbone
        self.classifier1 = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.classifier2 = classifier(in_features, num_classes, is_nonlinear, dropout)

        self.transfer_loss_weight = transfer_loss_weight
        self.num_mcd_steps = num_mcd_steps

        self.optimizer_backbone = torch.optim.AdamW(
            list(self.backbone.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.optimizer_classifiers = torch.optim.AdamW(
            list(self.classifier1.parameters()) + list(self.classifier2.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.log(torch.mean(predictions, dim=0) + 1e-6))

    def _classifier_discrepancy(
        self, predictions1: torch.Tensor, predictions2: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(torch.abs(predictions1 - predictions2))

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)

        p_s = 0.5 * (self.classifier1(f_s) + self.classifier2(f_s))
        p_t = 0.5 * (self.classifier1(f_t) + self.classifier2(f_t))

        return p_s, p_t

    def update(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
        ce_weight: torch.Tensor,
    ) -> dict:

        self.optimizer_backbone.zero_grad()
        self.optimizer_classifiers.zero_grad()

        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)

        y1_s = self.classifier1(f_s)
        y2_s = self.classifier2(f_s)
        y1_t = F.softmax(self.classifier1(f_t), dim=1)
        y2_t = F.softmax(self.classifier2(f_t), dim=1)

        cls_loss = (
            F.cross_entropy(y1_s, y_s, weight=ce_weight) +
            F.cross_entropy(y2_s, y_s, weight=ce_weight)
        )
        entropy_loss = (self._entropy(y1_t) + self._entropy(y2_t)) * 0.01

        step1_loss = cls_loss + entropy_loss
        step1_loss.backward()
        self.optimizer_backbone.step()
        self.optimizer_classifiers.step()

        self.optimizer_backbone.zero_grad()
        self.optimizer_classifiers.zero_grad()

        f_s = self.backbone(x_s).detach()
        f_t = self.backbone(x_t).detach()

        y1_s = self.classifier1(f_s)
        y2_s = self.classifier2(f_s)
        y1_t = F.softmax(self.classifier1(f_t), dim=1)
        y2_t = F.softmax(self.classifier2(f_t), dim=1)

        cls_loss = (
            F.cross_entropy(y1_s, y_s, weight=ce_weight) +
            F.cross_entropy(y2_s, y_s, weight=ce_weight)
        )
        entropy_loss = (self._entropy(y1_t) + self._entropy(y2_t)) * 0.01
        mcd_loss = self._classifier_discrepancy(y1_t, y2_t)

        step2_loss = cls_loss + entropy_loss - (self.transfer_loss_weight * mcd_loss)
        step2_loss.backward()
        self.optimizer_classifiers.step()

        for _ in range(self.num_mcd_steps):
            self.optimizer_backbone.zero_grad()

            f_t = self.backbone(x_t)
            y1_t = F.softmax(self.classifier1(f_t), dim=1)
            y2_t = F.softmax(self.classifier2(f_t), dim=1)

            mcd_loss = self._classifier_discrepancy(y1_t, y2_t)
            step3_loss = self.transfer_loss_weight * mcd_loss
            step3_loss.backward()
            self.optimizer_backbone.step()

        return {
            'cls_loss':     cls_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mcd_loss':     mcd_loss.item(),
            'total_loss':   step2_loss.item(),
        }

