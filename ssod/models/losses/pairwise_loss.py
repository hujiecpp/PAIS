import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
import torch.nn.functional as F

@LOSSES.register_module()
class PairwiseLoss(nn.Module):
    def __init__(self, loss_weight=1.0, pairwise_color_thresh=0.3):
        super(PairwiseLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pairwise_color_thresh = pairwise_color_thresh

    def forward(self, image_color_similarity, mask_pred, gt_bitmasks, flag):
        if flag == 1:
            pairwise_losses = compute_pairwise_term(mask_pred.unsqueeze(1))
            weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.unsqueeze(1).float()
            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            loss_pairwise *= self.loss_weight
        return loss_pairwise

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def compute_pairwise_term(mask_logits, pairwise_size=3, pairwise_dilation=2):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]