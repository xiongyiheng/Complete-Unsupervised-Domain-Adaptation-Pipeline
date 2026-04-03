import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


class ATDOC_NA(nn.Module):

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
        target_length: int,  # The total number of samples in the target training set
        warmup_steps: int,
        k_neighbors: int = 5,
    ):
        super(ATDOC_NA, self).__init__()

        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.transfer_loss_weight = transfer_loss_weight
        self.num_classes = num_classes
        self.warmup_steps = warmup_steps
        self.k_neighbors = k_neighbors

        self.register_buffer('update_count', torch.tensor(0))

        mem_fea = torch.rand(target_length, in_features)
        self.register_buffer('mem_fea', mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True))
        self.register_buffer('mem_cls', torch.ones(target_length, num_classes) / num_classes)

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
        tgt_index: torch.Tensor,  # The actual index in the target training set (accounts for permutation during training)
    ):
        self.update_count += 1

        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)

        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)

        cls_loss = F.cross_entropy(p_s, y_s, weight=ce_weight)
        transfer_loss = torch.tensor(0.0, device=x_s.device)

        if self.update_count > self.warmup_steps + 1:
            dis = -torch.mm(f_t.detach(), self.mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, tgt_index[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)

            w = torch.zeros(f_t.size(0), self.mem_fea.size(0), device=x_s.device)
            for wi in range(w.size(0)):
                for wj in range(self.k_neighbors):
                    w[wi][p1[wi, wj]] = 1.0 / self.k_neighbors

            weight_, pred = torch.max(w.mm(self.mem_cls), dim=1)
            loss_ = F.cross_entropy(p_t, pred, reduction='none')
            transfer_loss = torch.sum(weight_ * loss_) / torch.sum(weight_)

        total_loss = cls_loss + self.transfer_loss_weight * transfer_loss

        if self.update_count > self.warmup_steps:
            with torch.no_grad():
                f_t_fresh = self.backbone(x_t)
                p_t_fresh = self.classifier(f_t_fresh)
                f_t_norm = f_t_fresh / torch.norm(f_t_fresh, p=2, dim=1, keepdim=True)
                softmax_out = F.softmax(p_t_fresh, dim=1)
                sharpened = softmax_out ** 2 / (softmax_out ** 2).sum(dim=0)

            self.mem_fea[tgt_index] = f_t_norm.clone()
            self.mem_cls[tgt_index] = sharpened.clone()

        return total_loss, cls_loss, transfer_loss
