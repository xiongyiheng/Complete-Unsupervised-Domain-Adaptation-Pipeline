import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


def entropy(predictions: torch.Tensor, reduction: str) -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(predictions, reduction='none').detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions)
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


class MCC(nn.Module):
    def __init__(self, backbone, in_features, num_classes, is_nonlinear, dropout,
                 transfer_loss_weight, temperature):
        super(MCC, self).__init__()
        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.mcc_loss = MinimumClassConfusionLoss(temperature)
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
        p_t = self.classifier(f_t)

        cls_loss = F.cross_entropy(p_s, y_s, weight=ce_weight)
        
        transfer_loss = self.mcc_loss(p_t)

        total_loss = cls_loss + (self.transfer_loss_weight * transfer_loss)

        return total_loss, cls_loss, transfer_loss
