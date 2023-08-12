import torch
import numpy as np
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid_1
from mmdet.core.mask.structures import BitmapMasks
import torch.nn.functional as F
import datetime
import time


@DETECTORS.register_module()
class CoTKnet(MultiSteamDetector):
    def __init__(self, model1: dict, model2: dict, train_cfg=None, test_cfg=None):
        super(CoTKnet, self).__init__(
            dict(teacher=build_detector(model1), student=build_detector(model2)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
#------------------------------------------------------#
        #print("img.shape:", img.shape, img.device)
        # 假设按照 1:4 混合，img 的 shape 是 [5,3,h,w]
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        # 分成3组：有标签，无标签学生，无标签教师，每组都包括 img img_metas gt_bbox等
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            # 有标签分支正常训练
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
#---------------------------------------#            
            #print(data_groups["sup"]["img"])
            #data_groups["sup"]["img"] = data_groups["sup"]["img"].float()
            #print(data_groups["sup"]["img"])

            #print("sup_image:", data_groups["sup"]["img"].shape, data_groups["sup"]["img"].device)
            #start1 = time.time()
            stu_sup_loss = self.student.forward_train(**data_groups["sup"])
            tea_sup_loss = self.teacher.forward_train(**data_groups["sup"])
            #end1 = time.time()
            #print("sup_cost:", end1 - start1)
            stu_sup_loss = {"stu_sup_" + k: v for k, v in stu_sup_loss.items()}
            tea_sup_loss = {"tea_sup_" + k: v for k, v in tea_sup_loss.items()}
            loss.update(**stu_sup_loss)
            loss.update(**tea_sup_loss)
            
        if "unsup" in data_groups:
            # 无标签分支训练
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, data):
        
        loss={}
        
        img = data["img"]
        img_metas = data["img_metas"]        
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                img,
                img_metas
            )
            student_info = self.extract_student_info(
                img,
                img_metas
            )            
            
        x1 = self.student.extract_feat(img)
        x2 = self.teacher.extract_feat(img)
        
        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_labels = teacher_info["det_labels"]
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
#----------------------------------------#        
        #print("self.student.mask_assign_stride", self.student.mask_assign_stride)
        assign_H = pad_H // self.student.mask_assign_stride
        assign_W = pad_W // self.student.mask_assign_stride 
        
        for i, gt_mask in enumerate(teacher_info["det_masks"]):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)
                
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0]) 
                
        stu_gt_masks = gt_masks_tensor
        
        gt_masks_tensor = []
        gt_labels = student_info["det_labels"]
        #print("self.student.mask_assign_stride", self.student.mask_assign_stride)
        assign_H = pad_H // self.teacher.mask_assign_stride
        assign_W = pad_W // self.teacher.mask_assign_stride 
        
        for i, gt_mask in enumerate(student_info["det_masks"]):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)
                
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0]) 
                
        tea_gt_masks = gt_masks_tensor
        
        stu_rpn_results = self.student.rpn_head.forward_train(x1, img_metas, stu_gt_masks,
                                                  teacher_info["det_labels"])
        (stu_rpn_losses, stu_proposal_feats, stu_x_feats, stu_mask_preds,
         stu_cls_scores) = stu_rpn_results
#------------------------------------#
        #print("mask_preds.shape:", mask_preds.shape)
        stu_losses = self.student.roi_head.forward_train(
            stu_x_feats,
            stu_proposal_feats,
            stu_mask_preds,
            stu_cls_scores,
            img_metas,
            stu_gt_masks,
            teacher_info["det_labels"],
            imgs_whwh=None)

        stu_losses.update(stu_rpn_losses) 
        unsup_stu_losses = {"stu_" + k: v for k, v in stu_losses.items()}
        loss.update(**unsup_stu_losses)
        
        
        tea_rpn_results = self.teacher.rpn_head.forward_train(x1, img_metas, tea_gt_masks,
                                                  student_info["det_labels"])
        (tea_rpn_losses, tea_proposal_feats, tea_x_feats, tea_mask_preds,
         tea_cls_scores) = tea_rpn_results
#------------------------------------#
        #print("mask_preds.shape:", mask_preds.shape)
        tea_losses = self.teacher.roi_head.forward_train(
            tea_x_feats,
            tea_proposal_feats,
            tea_mask_preds,
            tea_cls_scores,
            img_metas,
            tea_gt_masks,
            student_info["det_labels"],
            imgs_whwh=None)

        tea_losses.update(tea_rpn_losses)         
        unsup_tea_losses = {"tea_" + k: v for k, v in tea_losses.items()}
        loss.update(**unsup_tea_losses)
        return loss


    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        #start2 = time.time()
        rpn_outs = self.student.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        #roi_outs = self.teacher.roi_head.simple_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        #num_roi_outs = len(roi_outs)
        seg_results, label_results, score_results = self.student.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        
        #end2 = time.time()
        #print("test_cost:", end2 - start2)
  
        #print("len(seg_results)", len(seg_results))
        #print("len(label_results)", len(label_results))
        #print("len(score_results)", len(score_results))
        #print("seg_results[0].masks.shape", seg_results[0].masks.shape)
        
       
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):    #过滤阈值去除分值比较低的检测框和类别
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
            
        # list(zip(*list)),将数组中的元组中的每一项取出,添加到一起,组成新的数组
#---------------------------------------------#        
        det_masks, det_labels, _ = list(   
            zip(
                *[
                    filter_invalid_1(
                        mask=seg_result,
                        label=label_result,
                        score=score_result,
                        thr=thr,
                    )
                    for seg_result, label_result, score_result in zip(
                        seg_results, label_results, score_results
                    )
                ]
            )
        )
       
        #print("det_masks[0].mask.shape:", det_masks[0].mask.shape)
        #print("det_labels[0].shape", det_labels[0].shape)
        #det_masks = seg_results
        #det_labels = label_results
        
        student_info["det_masks"] = det_masks
        student_info["det_labels"] = det_labels
        student_info["img_metas"] = img_metas
        return student_info
    
#！ 修改
#!!!!!!!!!
    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        #start2 = time.time()
        rpn_outs = self.teacher.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        seg_results, label_results, score_results = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        
        #end2 = time.time()
        #print("test_cost:", end2 - start2)
        #print("len(seg_results)", len(seg_results))
        #print("len(label_results)", len(label_results))
        #print("len(score_results)", len(score_results))
        #print("seg_results[0].masks.shape", seg_results[0].masks.shape)
        # filter invalid box roughly
        
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):    #过滤阈值去除分值比较低的检测框和类别
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
            
        # list(zip(*list)),将数组中的元组中的每一项取出,添加到一起,组成新的数组
        
        det_masks, det_labels, _ = list(   
            zip(
                *[
                    filter_invalid_1(
                        mask=seg_result,
                        label=label_result,
                        score=score_result,
                        thr=thr,
                    )
                    for seg_result, label_result, score_result in zip(
                        seg_results, label_results, score_results
                    )
                ]
            )
        )
        
#---------------------------------------------#        
        #print("det_masks[0].mask.shape:", det_masks[0].mask.shape)
        #print("det_labels[0].shape", det_labels[0].shape)
        #det_masks = seg_results
        #det_labels = label_results        
        
        teacher_info["det_masks"] = det_masks
        teacher_info["det_labels"] = det_labels
        teacher_info["img_metas"] = img_metas
        return teacher_info


    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
