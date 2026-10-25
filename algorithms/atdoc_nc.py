import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier
from .atdoc_na import label_smooth_ce


class ATDOC_NC(nn.Module):
    """
    ATDOC with a nearest-centroid memory (Liang et al., CVPR 2021), following the official training loop
    (github.com/tim-learn/ATDOC, demo_uda.py, --pl atdoc_nc):

      * source loss: label-smoothed CE (epsilon = 0.1), class-weighted;
      * target loss: CE to the nearest class centroid (cosine), ramped by eff = iter / max_iters;
      * centroids updated AFTER the optimizer step from the current target batch in eval mode, momentum 0.1.

    Training loop per iteration:
        loss, cls_loss, transfer_loss = model.compute_loss(x_s, y_s, x_t, ce_weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.update_memory(x_t)
    """

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
        max_iters: int,      # total training iterations (eff = iter / max_iters)
        momentum: float = 0.1,
        smooth: float = 0.1,
    ):
        super(ATDOC_NC, self).__init__()

        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.transfer_loss_weight = transfer_loss_weight
        self.num_classes = num_classes
        self.max_iters = max_iters
        self.momentum = momentum
        self.smooth = smooth

        self.register_buffer('iter_num', torch.tensor(0))

        mem_fea = torch.rand(num_classes, in_features)
        self.register_buffer('mem_fea', mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True))

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        p_s = self.classifier(self.backbone(x_s))
        p_t = self.classifier(self.backbone(x_t))
        return p_s, p_t

    def predict(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    def compute_loss(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
        ce_weight: torch.Tensor,
    ):
        self.iter_num += 1

        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)
        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)

        cls_loss = label_smooth_ce(p_s, y_s, self.num_classes, self.smooth, ce_weight)

        mem_fea_norm = self.mem_fea / torch.norm(self.mem_fea, p=2, dim=1, keepdim=True)
        dis = torch.mm(f_t.detach(), mem_fea_norm.t())
        _, pseudo_labels = torch.max(dis, dim=1)
        transfer_loss = F.cross_entropy(p_t, pseudo_labels)

        eff = self.iter_num.item() / self.max_iters
        total_loss = cls_loss + self.transfer_loss_weight * eff * transfer_loss

        return total_loss, cls_loss, transfer_loss

    @torch.no_grad()
    def update_memory(self, x_t: torch.Tensor):
        """Call after optimizer.step(): move the class centroids towards the current target batch (eval mode)."""
        was_training = self.training
        self.eval()
        f_t = self.backbone(x_t)
        softmax_t = F.softmax(self.classifier(f_t), dim=1)
        _, pred_t = torch.max(softmax_t, dim=1)
        onehot_t = torch.eye(self.num_classes, device=f_t.device)[pred_t]
        center_t = torch.mm(f_t.t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)
        self.mem_fea = (1.0 - self.momentum) * self.mem_fea + self.momentum * center_t.t()
        self.train(was_training)
