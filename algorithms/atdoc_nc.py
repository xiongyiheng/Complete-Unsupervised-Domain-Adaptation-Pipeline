import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


class ATDOC_NC(nn.Module):

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
    ):
        super(ATDOC_NC, self).__init__()

        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.transfer_loss_weight = transfer_loss_weight
        self.num_classes = num_classes

        mem_fea = torch.rand(num_classes, in_features)
        self.register_buffer('mem_fea', mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True))

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

        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)

        cls_loss = F.cross_entropy(p_s, y_s, weight=ce_weight)

        mem_fea_norm = self.mem_fea / torch.norm(self.mem_fea, p=2, dim=1, keepdim=True)
        dis = torch.mm(f_t.detach(), mem_fea_norm.t())
        _, pseudo_labels = torch.max(dis, dim=1)
        transfer_loss = F.cross_entropy(p_t, pseudo_labels)

        total_loss = cls_loss + self.transfer_loss_weight * transfer_loss

        with torch.no_grad():
            softmax_t = F.softmax(p_t.detach(), dim=1)
            _, pred_t = torch.max(softmax_t, dim=1)
            onehot_t = torch.eye(self.num_classes, device=f_t.device)[pred_t]
            center_t = torch.mm(f_t.detach().t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)
            self.mem_fea = (1.0 - 0.1) * self.mem_fea + 0.1 * center_t.t()

        return total_loss, cls_loss, transfer_loss
