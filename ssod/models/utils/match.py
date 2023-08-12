import numpy as np
import torch
import torch.nn as nn
from mmdet.core import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import MATCH_COST, build_match_cost
from mmcv.runner import force_fp32
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
    
    
@MATCH_COST.register_module(force=True)
class DiceCost(nn.Module):
    """DiceCost.

     Args:
         weight (int | float, optional): loss_weight
         pred_act (bool): Whether to activate the prediction
            before calculating cost

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 weight=1.,
                 pred_act=False,
                 act_mode='sigmoid',
                 eps=1e-3):
        super(DiceCost, self).__init__()
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode
        self.eps = eps

    def dice_loss(cls, input, target, eps=1e-3):
        input = input.reshape(input.size()[0], -1)
        target = target.reshape(target.size()[0], -1).float()
        # einsum saves 10x memory
        # a = torch.sum(input[:, None] * target[None, ...], -1)
        a = torch.einsum('nh,mh->nm', input, target)
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b[:, None] + c[None, ...])
        # 1 is a constance that will not affect the matching, so ommitted
        return -d

    @force_fp32(apply_to=('mask_preds', 'gt_masks'))
    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            mask_preds = mask_preds.sigmoid()
        elif self.pred_act:
            mask_preds = mask_preds.softmax(dim=0)
        dice_cost = self.dice_loss(mask_preds, gt_masks, self.eps)
        return dice_cost * self.weight


@MATCH_COST.register_module()
class MaskCost(nn.Module):
    """MaskCost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., pred_act=False, act_mode='sigmoid'):
        super(MaskCost, self).__init__()
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode
    
    @force_fp32(apply_to=('cls_pred', 'target'))
    def __call__(self, cls_pred, target):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            cls_pred = cls_pred.sigmoid()
        elif self.pred_act:
            cls_pred = cls_pred.softmax(dim=0)

        _, H, W = target.shape
        # flatten_cls_pred = cls_pred.view(num_proposals, -1)
        # eingum is ~10 times faster than matmul
        pos_cost = torch.einsum('nhw,mhw->nm', cls_pred, target)
        neg_cost = torch.einsum('nhw,mhw->nm', 1 - cls_pred, 1 - target)
        #neg_cost = torch.einsum('nhw,mhw->nm', (1 - cls_pred) / 1000, (1 - target) / 1000)
        
        
        cls_cost = -(pos_cost + neg_cost) / (H * W) 
        
        #print("-"*100)
        #print("cls_pred.dtype", cls_pred.dtype) #torch.float16
        #print("target.dtype", target.dtype) #torch.float16
        #print("cls_pred", cls_pred)
        #print("target", target)
        #print("pos_cost", pos_cost)
        #print("neg_cost", neg_cost) #会出现无穷大的情况
        #print("cls_cost", cls_cost)
        #print("h, w", H, W)
        return cls_cost * self.weight


#@BBOX_ASSIGNERS.register_module(force=True)
class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='FocalLossCost_1', weight=1.),
                 mask_cost=dict(type='SigmoidCost', weight=1.0),
                 dice_cost=dict(),
                 boundary_cost=None,
                 topk=1):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        if boundary_cost is not None:
            self.boundary_cost = build_match_cost(boundary_cost)
        else:
            self.boundary_cost = None
        self.topk = topk
        
        
    #@force_fp32(apply_to=('bbox_pred', 'cls_pred')) #TypeError: @force_fp32 can only be used to decorate the method of nn.Module
    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
            
            
            #print("+"*100)
            #print("cls_pred:", cls_pred) #nan torch.float16
            #print("gt_labels", gt_labels)
            #print(self.cls_cost)
            #print("cls_cost:", cls_cost)
            #print("cls_cost.dtype:", cls_cost.dtype) #torch.float32
            
        else:
            cls_cost = 0
        if self.mask_cost.weight != 0:
            reg_cost = self.mask_cost(bbox_pred, gt_bboxes)
            
            #print("+"*100)
            
            #print("reg_cost:", reg_cost)
            #print("self.mask_cost:", self.mask_cost) #MaskCost()
            #print("reg_cost.dtype:", reg_cost.dtype) #torch.float16
            
        else:
            reg_cost = 0
        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(bbox_pred, gt_bboxes)
            
            #print("+"*100)
            #print("dice_cost", dice_cost) 
            #print("dice_cost.dtype", dice_cost.dtype) #torch.float32
            
        else:
            dice_cost = 0
        if self.boundary_cost is not None and self.boundary_cost.weight != 0:
            b_cost = self.boundary_cost(bbox_pred, gt_bboxes)
        else:
            b_cost = 0
        
        #print("+"*100)
        #print("b_cost", b_cost) #0
        
        cost = cls_cost + reg_cost + dice_cost + b_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if self.topk == 1:
            
#-----------------------------------------#            
            #print("cost", cost)
            #print(cost.shape)
        
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        else:
            topk_matched_row_inds = []
            topk_matched_col_inds = []
            for i in range(self.topk):
                matched_row_inds, matched_col_inds = linear_sum_assignment(
                    cost)
                topk_matched_row_inds.append(matched_row_inds)
                topk_matched_col_inds.append(matched_col_inds)
                cost[matched_row_inds] = 1e10
            matched_row_inds = np.concatenate(topk_matched_row_inds)
            matched_col_inds = np.concatenate(topk_matched_col_inds)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
