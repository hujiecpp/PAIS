import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet.core.mask.structures import BitmapMasks
from torch.nn import functional as F

def filter_invalid_bbox_iou(bbox, label=None, score=None, iou=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        #valid2 = iou > iou_thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        
        bbox = bbox[valid]
        if iou is not None:
            iou = iou[valid]
        
        
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            """
            print("------------------")
            print("bbox:", bbox)
            print(bbox.shape)
            print(valid)
            print(valid.shape)
            print("mask:", mask)
            print(mask.shape)
            """
            #mask = mask[valid]
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            
            
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        
        if iou is not None:
            iou = iou[valid]
        
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return bbox, label, iou, mask


def filter_invalid_mask_iou(bbox, label=None, score=None, bbox_iou=None, mask=None, mask_iou=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        #valid2 = iou > iou_thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        
        bbox = bbox[valid]
        if bbox_iou is not None:
            bbox_iou = bbox_iou[valid]
        if mask_iou is not None:
            mask_iou = mask_iou[valid]
        
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            """
            print("------------------")
            print("bbox:", bbox)
            print(bbox.shape)
            print(valid)
            print(valid.shape)
            print("mask:", mask)
            print(mask.shape)
            """
            #mask = mask[valid]
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            
            
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        
        if bbox_iou is not None:
            bbox_iou = bbox_iou[valid]
            
        if mask_iou is not None:
            mask_iou = mask_iou[valid]
            
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return bbox, label, bbox_iou, mask, mask_iou