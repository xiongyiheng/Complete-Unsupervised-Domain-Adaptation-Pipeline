import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import classifier


def label_smooth_ce(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, epsilon: float,
                    class_weight: torch.Tensor = None) -> torch.Tensor:
    """Official ATDOC source loss: label-smoothed CE per sample (CrossEntropyLabelSmooth, reduction='none'),
    averaged with per-sample class weights: sum(w[y] * l) / sum(w[y])."""
    log_probs = F.log_softmax(logits, dim=1)
    targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
    targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1)
    if class_weight is None:
        return loss.mean()
    w = class_weight[labels]
    return torch.sum(w * loss) / torch.sum(w).item()


class ATDOC_NA(nn.Module):
    """
    ATDOC with neighborhood aggregation (Liang et al., CVPR 2021), following the official training loop
    (github.com/tim-learn/ATDOC, demo_uda.py, --pl atdoc_na):

      * source loss: label-smoothed CE (epsilon = 0.1), class-weighted;
      * target loss: CE to the pseudo label of the K nearest memory neighbours (self excluded), weighted by the
        neighbours' aggregated confidence, ramped by eff = iter / max_iters from the first iteration;
      * memory (normalised features + sharpened predictions of every target training sample) is refreshed for the
        current batch AFTER the optimizer step, with the network in eval mode (momentum 1.0 = replace).

    Training loop per iteration:
        loss, cls_loss, transfer_loss = model.compute_loss(x_s, y_s, x_t, ce_weight, tgt_index)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.update_memory(x_t, tgt_index)
    tgt_index = position of each x_t in the target training set (its loader must return the index).
    """

    def __init__(
        self,
        backbone,
        in_features: int,
        num_classes: int,
        is_nonlinear: bool,
        dropout: float,
        transfer_loss_weight: float,
        target_length: int,  # The total number of samples in the target training set
        max_iters: int,      # total training iterations (eff = iter / max_iters)
        k_neighbors: int = 5,
        momentum: float = 1.0,
        smooth: float = 0.1,
    ):
        super(ATDOC_NA, self).__init__()

        self.backbone = backbone
        self.classifier = classifier(in_features, num_classes, is_nonlinear, dropout)
        self.transfer_loss_weight = transfer_loss_weight
        self.num_classes = num_classes
        self.max_iters = max_iters
        self.k_neighbors = k_neighbors
        self.momentum = momentum
        self.smooth = smooth

        self.register_buffer('iter_num', torch.tensor(0))

        mem_fea = torch.rand(target_length, in_features)
        self.register_buffer('mem_fea', mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True))
        self.register_buffer('mem_cls', torch.ones(target_length, num_classes) / num_classes)

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
        tgt_index: torch.Tensor,  # The actual index in the target training set (accounts for permutation during training)
    ):
        self.iter_num += 1

        f_s = self.backbone(x_s)
        f_t = self.backbone(x_t)
        p_s = self.classifier(f_s)
        p_t = self.classifier(f_t)

        cls_loss = label_smooth_ce(p_s, y_s, self.num_classes, self.smooth, ce_weight)

        dis = -torch.mm(f_t.detach(), self.mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, tgt_index[di]] = torch.max(dis)  # never pick the sample itself
        _, p1 = torch.sort(dis, dim=1)

        w = torch.zeros(f_t.size(0), self.mem_fea.size(0), device=f_t.device)
        w.scatter_(1, p1[:, :self.k_neighbors], 1.0 / self.k_neighbors)

        weight_, pred = torch.max(w.mm(self.mem_cls), dim=1)
        loss_ = F.cross_entropy(p_t, pred, reduction='none')
        transfer_loss = torch.sum(weight_ * loss_) / torch.sum(weight_).item()

        eff = self.iter_num.item() / self.max_iters
        total_loss = cls_loss + self.transfer_loss_weight * eff * transfer_loss

        return total_loss, cls_loss, transfer_loss

    @torch.no_grad()
    def update_memory(self, x_t: torch.Tensor, tgt_index: torch.Tensor):
        """Call after optimizer.step(): refresh the memory of the current target batch in eval mode."""
        was_training = self.training
        self.eval()
        f_t = self.backbone(x_t)
        p_t = self.classifier(f_t)
        f_t = f_t / torch.norm(f_t, p=2, dim=1, keepdim=True)
        softmax_out = F.softmax(p_t, dim=1)
        sharpened = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))
        self.mem_fea[tgt_index] = (1.0 - self.momentum) * self.mem_fea[tgt_index] + self.momentum * f_t
        self.mem_cls[tgt_index] = (1.0 - self.momentum) * self.mem_cls[tgt_index] + self.momentum * sharpened
        self.train(was_training)
