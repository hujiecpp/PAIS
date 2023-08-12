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
class Robust_e2e(MultiSteamDetector):
    def __init__(self, model1: dict, model2: dict, train_cfg=None, test_cfg=None):
        super(Robust_e2e, self).__init__(
            dict(teacher1=build_detector(model1), teacher2=build_detector(model2), student1=build_detector(model1), student2=build_detector(model2)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher1")
            self.freeze("teacher2")
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
            sup_m1_loss = self.student1.forward_train(**data_groups["sup"])
            sup_m2_loss = self.student2.forward_train(**data_groups["sup"])
            #end1 = time.time()
            #print("sup_cost:", end1 - start1)
            sup_m1_loss = {"sup_m1_" + k: v for k, v in sup_m1_loss.items()}
            sup_m2_loss = {"sup_m2_" + k: v for k, v in sup_m2_loss.items()}
            loss.update(**sup_m1_loss)
            loss.update(**sup_m2_loss)
            
        if "unsup_student" in data_groups:
            # 无标签分支训练
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    
    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        #！注意，这边的输入都是batch个数据，所以image_metas是多个图片的meta集合
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        
        with torch.no_grad():
            teacher_info, teacher_info_1 = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx]
            )
        
        return self.compute_pseudo_label_loss(student_data["img"], student_data["img_metas"],  teacher_info, teacher_info_1)
    
    
    def compute_pseudo_label_loss(self, img, img_metas, teacher_info, teacher_info_1):
        
        loss={}
        
        x1 = self.student1.extract_feat(img)
        x2 = self.student2.extract_feat(img)
        student_transform_matrix = [
            torch.from_numpy(meta["transform_matrix"]).float().to(x1[0][0].device)
            for meta in img_metas
        ]

        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_transform_matrix
        )
#！由于两个通道的数据增强不一致，需要将teacher的mask变换到student上，mask的变换----------------------#
        pseudo_masks = self._transform_mask(
            teacher_info["det_masks"],   #list(bitmapmasks)    3, 
            M,
            [meta["img_shape"] for meta in img_metas],
        )
        t1_gt_labels = teacher_info["det_labels"]        
        t1_gt_masks = pseudo_masks
              
        pseudo_masks_1 = self._transform_mask(
            teacher_info_1["det_masks"],   #list(bitmapmasks)    3, 
            M,
            [meta["img_shape"] for meta in img_metas],
        )
        t2_gt_labels = teacher_info_1["det_labels"] 
        t2_gt_masks = pseudo_masks_1           

        
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        
#将tea1得到的伪标签用于stu2
        assign_H = pad_H // self.student2.mask_assign_stride
        assign_W = pad_W // self.student2.mask_assign_stride
        gt_masks_tensor = []
        for i, gt_mask in enumerate(t1_gt_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, t1_gt_labels[0].device)
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
        
        t1_gt_masks = gt_masks_tensor
#----------------------------------#
#这边有问题，应该是x2
        t1_rpn_results = self.student2.rpn_head.forward_train(x1, img_metas, t1_gt_masks,
                                                  t1_gt_labels)
        (t1_rpn_losses, t1_proposal_feats, t1_x_feats, t1_mask_preds,
         t1_cls_scores) = t1_rpn_results
        
        t1_loss_rpn_seg = t1_rpn_losses['loss_rpn_seg']
        t1_rpn_losses['loss_rpn_seg'] = t1_loss_rpn_seg * 0        
        t1_losses = self.student2.roi_head.forward_train(
            t1_x_feats,
            t1_proposal_feats,
            t1_mask_preds,
            t1_cls_scores,
            img_metas,
            t1_gt_masks,
            t1_gt_labels,
            imgs_whwh=None)        
        t1_losses.update(t1_rpn_losses)
        t1_losses = weighted_loss(t1_losses, weight=0.6)
        unsup_t1_losses = {"t1_" + k: v for k, v in t1_losses.items()}
        loss.update(**unsup_t1_losses)
        

#将tea2得到的伪标签用于stu1
        assign_H = pad_H // self.student1.mask_assign_stride
        assign_W = pad_W // self.student1.mask_assign_stride
        gt_masks_tensor = []
        for i, gt_mask in enumerate(t2_gt_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, t2_gt_labels[0].device)
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
        
        t2_gt_masks = gt_masks_tensor
#----------------------------------#
#这边有问题，应该是x1
        t2_rpn_results = self.student1.rpn_head.forward_train(x2, img_metas, t2_gt_masks,
                                                  t2_gt_labels)
        (t2_rpn_losses, t2_proposal_feats, t2_x_feats, t2_mask_preds,
         t2_cls_scores) = t2_rpn_results
        
        t2_loss_rpn_seg = t2_rpn_losses['loss_rpn_seg']
        t2_rpn_losses['loss_rpn_seg'] = t2_loss_rpn_seg * 0        
        t2_losses = self.student1.roi_head.forward_train(
            t2_x_feats,
            t2_proposal_feats,
            t2_mask_preds,
            t2_cls_scores,
            img_metas,
            t2_gt_masks,
            t2_gt_labels,
            imgs_whwh=None)        
        t2_losses.update(t2_rpn_losses)
        t2_losses = weighted_loss(t2_losses, weight=0.6)
        unsup_t2_losses = {"t2_" + k: v for k, v in t2_losses.items()}
        loss.update(**unsup_t2_losses)
        
        return loss


    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        feat = self.student1.extract_feat(img)
        student_info["backbone_feature"] = feat
        #start2 = time.time()
        rpn_outs = self.student1.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        seg_results, label_results, score_results = self.student1.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        """
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float): 
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")     
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
        """
        det_masks = seg_results
        det_labels = label_results
        student_info["det_masks"] = det_masks
        student_info["det_labels"] = det_labels
        student_info["img_metas"] = img_metas
        
        student_info_1 = {}
        feat1 = self.student2.extract_feat(img)
        student_info_1["backbone_feature"] = feat1
        rpn_outs1 = self.student2.rpn_head.simple_test_rpn(feat1, img_metas)
        (proposal_feats1, x_feats1, mask_preds1, cls_scores1, seg_preds1) = rpn_outs1
        seg_results1, label_results1, score_results1 = self.student2.roi_head.teacher_test(x_feats1, proposal_feats1, mask_preds1, cls_scores1, img_metas)
        det_masks1 = seg_results1
        det_labels1 = label_results1
        student_info_1["det_masks"] = det_masks1
        student_info_1["det_labels"] = det_labels1
        student_info_1["img_metas"] = img_metas        
        return student_info, student_info_1
    
#！ 修改
#!!!!!!!!!
    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher1.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        rpn_outs = self.teacher1.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        seg_results, label_results, score_results = self.teacher1.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float): 
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            raise NotImplementedError("Dynamic Threshold is not implemented yet.") 
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
        teacher_info["det_masks"] = det_masks
        teacher_info["det_labels"] = det_labels
        teacher_info["img_metas"] = img_metas
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]        
        
        
        teacher_info_1 = {}
        feat1 = self.teacher2.extract_feat(img)
        teacher_info_1["backbone_feature"] = feat1
        rpn_outs1 = self.teacher2.rpn_head.simple_test_rpn(feat1, img_metas)
        (proposal_feats1, x_feats1, mask_preds1, cls_scores1, seg_preds1) = rpn_outs1
        seg_results1, label_results1, score_results1 = self.teacher2.roi_head.teacher_test(x_feats1, proposal_feats1, mask_preds1, cls_scores1, img_metas)
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float): 
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            raise NotImplementedError("Dynamic Threshold is not implemented yet.") 
        det_masks1, det_labels1, _ = list(   
            zip(
                *[
                    filter_invalid_1(
                        mask=seg_result,
                        label=label_result,
                        score=score_result,
                        thr=thr,
                    )
                    for seg_result, label_result, score_result in zip(
                        seg_results1, label_results1, score_results1
                    )
                ]
            )
        )  
        teacher_info_1["det_masks"] = det_masks1
        teacher_info_1["det_labels"] = det_labels1
        teacher_info_1["img_metas"] = img_metas   
        teacher_info_1["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat1[0][0].device)
            for meta in img_metas
        ]
        
        return teacher_info, teacher_info_1

    
    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes
    
    
#！加一个函数，类似于_transform_bbox，用于mask的变换
    #@force_fp32(apply_to=["masks", "trans_mat"])
    def _transform_mask(self, masks, trans_mat, max_shape):
        masks = Transform2D.transform_masks(masks, trans_mat, max_shape)
        return masks
#-------------------------------------------------#

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]    
    

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
        if not any(["student1" in key or "teacher1" in key or "teacher2" in key or "student2" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher1." + k: state_dict[k] for k in keys})
            state_dict.update({"teacher2." + k: state_dict[k] for k in keys})
            state_dict.update({"student1." + k: state_dict[k] for k in keys})
            state_dict.update({"student2." + k: state_dict[k] for k in keys})
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
