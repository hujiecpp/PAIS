import torch
import numpy as np
from mmcv.ops.nms import batched_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps

def compute_single_bbox_iou(pred_boxes, gt_box):
	# first step
	# inter left up point and right bottom point
	left_up_x=max(pred_boxes[0], gt_box[0])
	left_up_y=max(pred_boxes[1], gt_box[1])
	right_bottom_x=min(pred_boxes[2], gt_box[2])
	right_bottom_y=min(pred_boxes[3], gt_box[3])
	# second step
	# intersect area calculate
	intersect=(right_bottom_x - left_up_x) * (right_bottom_y - left_up_y)

	# third step
	# union area calculate
	area_pred=(pred_boxes[2]-pred_boxes[0]) * (pred_boxes[3]-pred_boxes[1])
	area_gt=(gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
	union=(area_pred + area_gt - intersect)

	# four step
	# iou= inter /union
	iou = intersect / (union + 1e-6)
	return iou


def compute_bbox_iou(b1, b2):
    assert b1.shape[0] == b2.shape[0]
    num_bbox = b1.shape[0]
    iou_lst = []
    for i in range(num_bbox):
        iou_lst.append(compute_single_bbox_iou(b1[i], b2[i]))
    iou_tensor = torch.tensor(iou_lst).to(b1.device)
    return iou_tensor


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding 
    binarized_inputs = (inputs >= 0.5).float()
    #print("binarized_inputs", binarized_inputs)
    #targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


#提供iou的nms
def multiclass_nms_iou(multi_bboxes,
                   multi_scores,
                   iou_score,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    
    '''
    print("+"*100)
    print("+"*100)
    print("bboxes", bboxes.shape)
    print("scores", scores.shape)
    print("scores", scores.shape) #torch.Size([80000])
    print("labels", labels.shape)
    print("bboxes", bboxes.detach().cpu().numpy())
    '''
    
    
    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        '''
        print("-"*100)
        print("-"*100)
        print("inds", inds)
        '''
        inds_iou = inds // num_classes
        #print("inds_iou", inds_iou)
        iou_score = iou_score[inds_iou]
        
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
        iou_score = torch.cat([iou_score, iou_score.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels, iou_score
    ''' #这边已经出错了
    print("+"*100)
    print("+"*100)
    print("bboxes", bboxes.shape)
    print("iou_score", iou_score.shape)
    print("bboxes", bboxes.detach().cpu().numpy())
    '''
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    
    
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
        
        
    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        #print("keep", keep)
        keep1 = keep.to(iou_score.device)
        return dets, labels[keep], iou_score[keep1]
    
    
    
def bbox2result_iou(bboxes, labels, ious, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)], [np.zeros((0, 1), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            '''
            print("-"*100)
            print("-"*100)
            print("bbox", bboxes)
            '''
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            ious = ious.detach().cpu().numpy()
        iou_result = [[] for _ in range(num_classes)]
        num_ins = bboxes.shape[0]
        bbox_result = [bboxes[labels == i, :] for i in range(num_classes)]
        for idx in range(num_ins):
            iou_result[labels[idx]].append(ious[idx])
        
        return bbox_result, iou_result