import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional

from .classifier import classifier 

def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:

    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix

class GaussianKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = nn.ModuleList(kernels) 
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])

        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss

class MMD(nn.Module):
    def __init__(self, backbone, in_features, num_classes, is_nonlinear, dropout, transfer_loss_weight):
        super(MMD, self).__init__()
        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        
        kernels = [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels, linear=False)
        
        self.transfer_loss_weight = transfer_loss_weight

    def forward(self, x_s, x_t):
        f_s = self.backbone(x_s)
        f_t = self.backgone(x_t)
        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)
        return p_s, p_t

    def compute_loss(self, x_s, y_s, x_t, ce_weight):
        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)
        
        p_s = self.classifier(f_s)
        cls_loss = F.cross_entropy(p_s, y_s, ce_weight)
        
        transfer_loss = self.mkmmd_loss(f_s, f_t)
        
        total_loss = cls_loss + (self.transfer_loss_weight * transfer_loss)
        
        return total_loss, cls_loss, transfer_loss

