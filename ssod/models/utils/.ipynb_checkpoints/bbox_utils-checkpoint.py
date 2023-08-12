import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet.core.mask.structures import BitmapMasks
from torch.nn import functional as F

def filter_invalid_stage1(bbox=None, label=None, score=None, mask=None, mask_score=None, iou=None, thr=0.0, min_size=None):
    if score is not None:
        valid = score > thr

        
        if bbox is not None:
            bbox = bbox[valid]
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
        if iou is not None:
            iou = iou[valid]
        if score is not None:
            score = score[valid]
        if mask_score is not None:
            mask_score = mask_score[valid]
            
    if min_size is not None:
        if bbox is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return mask, mask_score, label, score, iou

def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    check_is_tensor(dst_pix_trans_src_pix)

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(
                dst_pix_trans_src_pix.shape
            )
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix.float()).to(
        src_norm_trans_src_pix.dtype
    )
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError(
            "Input src type is not a torch.Tensor. Got {}".format(type(src))
        )

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpAffine."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), dsize)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm.float())

    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners,
    )

    return F.grid_sample(
        src.float(),
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    ).to(src.dtype)


class Transform2D:
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            
#!-------------------------------------------------#            
            """
            print("+-+-+-+_+_+_+_+_+_+_+_+")
            print(len(bbox))   
            print(type(bbox))    #list
            print(bbox[0])       #tensor
            print("//////////////////////")
            print(len(M))
            print(type(M))
            """
            #print(M.shape)
#
            
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            
#!-------------------------------------------------#            
            """
            print("+-+-+-+_+_+_+_+_+_+_+_+")
            print(len(bbox))   
            print(type(bbox))    #Tensor
            #print(bbox[0])
            print("//////////////////////")
            print(len(M))
            print(M)
            print(type(M))
            """
            #print(M.shape)
#
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],  #共用声明和共用一变量定义
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            
#!-------------------------------------------------#            
            """
            print("++++++++++++++++++++++++++++++")
            print(len(mask))   
            print(type(mask))    #list
            print(type(mask[0]))  #list,实际上这边应该是BitmapMasks
            print("-----------------------------")
            print(len(M))
            print(type(M))
            """
            #print(M.shape)
            
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            
            #print("++++++++++++++++++++++++++++++")
            #print("mask:", mask)
            #print(len(mask))   
            #print(type(mask))    #list
            #print(type(mask[0]))  #list,实际上这边应该是BitmapMasks
            #print("-----------------------------")
            #print(len(M))
            #print(type(M))
            #print(*out_shape)
            
            if mask.masks.shape[0] == 0:
                #return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
                return BitmapMasks(np.zeros((0, out_shape[0], out_shape[1])), out_shape[0], out_shape[1])
            
            #print("out_shape:", out_shape)
            #print("mask.masks[:, None, ...]:", mask.masks[:, None, ...])
            #print(mask.masks[:, None, ...].shape)
            
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None, ...]).to(M.device).to(M.dtype)
            )
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),        #None在tensor中的作用类似torch.unsqueeze(),方便扩展维度，而不改变数据排列顺序。
                    #out_shape,
                    (out_shape[0], out_shape[1]),
                )
                .squeeze(1)
                .cpu()
                .numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], (out_shape[0], out_shape[1]), mode="nearest")
                .squeeze()
                .to(img.dtype)
            )


#!修改mask部分----------------------------#
#----------------------------------------#
def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        
        bbox = bbox[valid]
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
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return bbox, label, mask


def filter_invalid_1(bbox=None, label=None, score=None, mask=None, thr=0.0, min_size=None):
    if score is not None:
        valid = score > thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        
        #print("valid:", valid)
        #print("valid.shape:", valid.shape)
        #print("label:", label)
        #print("label.shape:", label.shape)
        if score is not None:
            score = score[valid]
        if bbox is not None:
            bbox = bbox[valid]
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
        if bbox is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return mask, label, score


def filter_invalid_2(bbox=None, label=None, score=None, mask=None, iou=None, thr=0.0, iou_thr=0.7, min_size=None):
    if score is not None:
        valid_0 = score > thr
        valid_1 = iou > iou_thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        valid = valid_0 * valid_1
        #print("valid:", valid)
        #print("valid.shape:", valid.shape)
        #print("label:", label)
        #print("label.shape:", label.shape)
        
        if bbox is not None:
            bbox = bbox[valid]
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
        if bbox is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return mask, label, bbox


def filter_invalid_3(bbox=None, label=None, score=None, mask=None, iou=None, thr=0.0, iou_thr=0.7, min_size=None):
    if score is not None:
        valid_0 = score > thr
        valid_1 = iou > iou_thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        valid = valid_0 * valid_1
        #print("valid:", valid)
        #print("valid.shape:", valid.shape)
        #print("label:", label)
        #print("label.shape:", label.shape)
        
        if bbox is not None:
            bbox = bbox[valid]
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
        if iou is not None:
            iou = iou[valid]
        if score is not None:
            score = score[valid]
            
    if min_size is not None:
        if bbox is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return mask, label, score, iou


def filter_invalid_4(bbox=None, label=None, score=None, mask=None, iou=None, mask_thr=None, thr=0.0, iou_thr=0.7, min_size=None):
    if score is not None:
        valid_0 = score > thr
        valid_1 = iou > iou_thr
        #print("///////////////////////")
        #print("原来的bbox:", bbox)
        #print(bbox.shape)
        valid = valid_0 * valid_1
        #print("valid:", valid)
        #print("valid.shape:", valid.shape)
        #print("label:", label)
        #print("label.shape:", label.shape)
        
        if bbox is not None:
            bbox = bbox[valid]
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
        if iou is not None:
            iou = iou[valid]
        if score is not None:
            score = score[valid]
        
        if mask_thr is not None:
            mask_thr = mask_thr[valid]
            
    if min_size is not None:
        if bbox is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
            #mask = mask[valid]
            
    return mask, label, score, iou, mask_thr

class Transform2D_2:
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            
#!-------------------------------------------------#            
            """
            print("+-+-+-+_+_+_+_+_+_+_+_+")
            print(len(bbox))   
            print(type(bbox))    #list
            print(bbox[0])       #tensor
            print("//////////////////////")
            print(len(M))
            print(type(M))
            """
            #print(M.shape)
#
            
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            
#!-------------------------------------------------#            
            """
            print("+-+-+-+_+_+_+_+_+_+_+_+")
            print(len(bbox))   
            print(type(bbox))    #Tensor
            #print(bbox[0])
            print("//////////////////////")
            print(len(M))
            print(M)
            print(type(M))
            """
            #print(M.shape)
#
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],  #共用声明和共用一变量定义
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            
#!-------------------------------------------------#            
            """
            print("++++++++++++++++++++++++++++++")
            print(len(mask))   
            print(type(mask))    #list
            print(type(mask[0]))  #list,实际上这边应该是BitmapMasks
            print("-----------------------------")
            print(len(M))
            print(type(M))
            """
            #print(M.shape)
            
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            
            #print("++++++++++++++++++++++++++++++")
            #print("mask:", mask)
            #print(len(mask))   
            #print(type(mask))    #list
            #print(type(mask[0]))  #list,实际上这边应该是BitmapMasks
            #print("-----------------------------")
            #print(len(M))
            #print(type(M))
            #print(*out_shape)
            
            if mask.masks.shape[0] == 0:
                #return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
                return BitmapMasks(np.zeros((0, out_shape[0], out_shape[1])), out_shape[0], out_shape[1])
            
            #print("out_shape:", out_shape)
            #print("mask.masks[:, None, ...]:", mask.masks[:, None, ...])
            #print(mask.masks[:, None, ...].shape)
            
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None, ...]).to(M.device).to(M.dtype)
            )
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),        #None在tensor中的作用类似torch.unsqueeze(),方便扩展维度，而不改变数据排列顺序。
                    #out_shape,
                    (out_shape[0], out_shape[1]),
                )
                .squeeze(1)
                .cpu()
                .numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], (out_shape[0], out_shape[1]), mode="nearest")
                .squeeze()
                .to(img.dtype)
            )
        
class Transform2D_1:
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            
#!-------------------------------------------------#            
            """
            print("+-+-+-+_+_+_+_+_+_+_+_+")
            print(len(bbox))   
            print(type(bbox))    #list
            print(bbox[0])       #tensor
            print("//////////////////////")
            print(len(M))
            print(type(M))
            """
            #print(M.shape)
#
            
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            
#!-------------------------------------------------#            
            """
            print("+-+-+-+_+_+_+_+_+_+_+_+")
            print(len(bbox))   
            print(type(bbox))    #Tensor
            #print(bbox[0])
            print("//////////////////////")
            print(len(M))
            print(M)
            print(type(M))
            """
            #print(M.shape)
#
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],  #共用声明和共用一变量定义
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            
#!-------------------------------------------------#            
            """
            print("++++++++++++++++++++++++++++++")
            print(len(mask))   
            print(type(mask))    #list
            print(type(mask[0]))  #list,实际上这边应该是BitmapMasks
            print("-----------------------------")
            print(len(M))
            print(type(M))
            """
            #print(M.shape)
            
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            
            #print("++++++++++++++++++++++++++++++")
            #print("mask:", mask)
            #print(len(mask))   
            #print(type(mask))    #list
            #print(type(mask[0]))  #list,实际上这边应该是BitmapMasks
            #print("-----------------------------")
            #print(len(M))
            #print(type(M))
            #print(*out_shape)
            
            if mask.masks.shape[0] == 0:
                #return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
                return BitmapMasks(np.zeros((0, out_shape[0], out_shape[1])), out_shape[0], out_shape[1])
            
            #print("out_shape:", out_shape)
            #print("mask.masks[:, None, ...]:", mask.masks[:, None, ...])
            #print(mask.masks[:, None, ...].shape)
            
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None, ...]).to(M.device).to(M.dtype)
            )
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),        #None在tensor中的作用类似torch.unsqueeze(),方便扩展维度，而不改变数据排列顺序。
                    #out_shape,
                    (out_shape[0], out_shape[1]),
                )
                .squeeze(1)
                .cpu()
                .numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], (out_shape[0], out_shape[1]), mode="nearest")
                .squeeze()
                .to(img.dtype)
            )