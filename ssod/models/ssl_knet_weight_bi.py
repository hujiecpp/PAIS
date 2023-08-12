import torch
import numpy as np
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, mask_matrix_nms
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n, log_image_with_masks, log_image_with_masks_without_box, isVisualbyCount, log_image_with_boxes, color_transform

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid_3
from mmdet.core.mask.structures import BitmapMasks
import torch.nn.functional as F
import datetime
import time
import matplotlib as mpl
from skimage import color
mpl.use('Agg')

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

def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


@DETECTORS.register_module()
class SslKnet_weight_bi(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SslKnet_weight_bi, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
        #self.sum_psedudo_mask = torch.tensor(0, device="cuda").float()
        self.sum_psedudo_mask = 0
        #self.sum_none_box = torch.tensor(0, device="cuda").float()
        self.sum_none_box = 0
        self.pairwise_size = self.train_cfg.pairwise_size
        self.pairwise_dilation = self.train_cfg.pairwise_dilation

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
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
            '''
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            '''
#---------------------------------------# 
            #print("data_groups['sup']", data_groups["sup"]) #img_metas中有sup属性
    
            #print(data_groups["sup"]["img"])
            #data_groups["sup"]["img"] = data_groups["sup"]["img"].float()
            #print(data_groups["sup"]["img"])
            #print(data_groups["sup"]["img"].shape)
            #start1 = time.time()
            sup_loss = self.student.forward_train(**data_groups["sup"])
            """
            iou_loss = sup_loss["s0_loss_iou"]
            sup_loss["s0_loss_iou"] = iou_loss * 0
            iou_loss = sup_loss["s1_loss_iou"]
            sup_loss["s1_loss_iou"] = iou_loss * 0        
            iou_loss = sup_loss["s2_loss_iou"]
            sup_loss["s2_loss_iou"] = iou_loss * 0 
            """
            #end1 = time.time()
            #print("sup_cost:", end1 - start1)
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
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
        
        #start = time.time()
        #starttime = datetime.datetime.now() 
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        #student_info = self.extract_student_info(**student_data)
        #endtime = datetime.datetime.now() 
        #end = time.time()
        #print("cost_time:", endtime - starttime) #需要cost_time: 0:00:05.281560
        #print("time:", end - start)

        return self.compute_pseudo_label_loss(student_data["img"], student_data["img_metas"],  teacher_info)

#!修改
#-----------
    def compute_pseudo_label_loss(self, img, img_metas, teacher_info):
        
        x = self.student.extract_feat(img)
        
        student_transform_matrix = [
            torch.from_numpy(meta["transform_matrix"]).float().to(x[0][0].device)
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
        gt_labels = teacher_info["det_labels"]
        
#将mask变换到原图大小，并在原图尺度上可视化出来
        interval = 1000
        flag = isVisualbyCount(interval)
        if flag == 1:

            M1 = [at.inverse() for at in teacher_info["transform_matrix"]]
            M2 = [at.inverse() for at in student_transform_matrix]

            mask_ori = self._transform_mask(
                teacher_info["det_masks"],   #list(bitmapmasks)    3, 
                M1,
                [meta["ori_shape"] for meta in img_metas],
            )

            for i in range(len(img_metas)):
                #img_ori = cv2.imread(img_metas[i]['filename'])
                #img_ori = torch.tensor(img_ori).cpu()
                img_ori = Transform2D.transform_image(img[i], M2[i], img_metas[i]["ori_shape"])
                img_ori = img_ori.cpu().detach()
                #print("img_ori.shape", img_ori.shape)
                mask_vis = mask_ori[i].to_tensor(torch.float, img[0].device).cpu().detach()
                mask_vis = mask_vis > 0.5
                label_vis = gt_labels[i].cpu().detach()
                if  mask_vis.shape[0] > 0:
                    log_image_with_masks_without_box(
                            "mask_ori",
                            img_ori,
                            None,
                            mask_vis,
                            bbox_tag="mask_ori",
                            labels=label_vis,
                            class_names=self.CLASSES,
                            interval=1,
                            img_norm_cfg=img_metas[i]["img_norm_cfg"],
                        )        
        
        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_soft_masks_tensor = []
        gt_bbox_tensor = []
        gt_box_mask_tensor = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        
#----------------------------------------#        
        #print("self.student.mask_assign_stride", self.student.mask_assign_stride)
        
        assign_H = pad_H // self.student.mask_assign_stride
        assign_W = pad_W // self.student.mask_assign_stride

        #levelset_img = F.interpolate(img, (assign_H, assign_W), 
        #mode='bilinear', align_corners=False)
        
#------------------------------------------#        
        #print("gt_masks[0].shape:", len(gt_masks), gt_masks[0].masks.shape)   #2 (14, 800, 1088)
        #print("gt_semantic_seg[0].shape:", len(gt_semantic_seg), gt_semantic_seg[0].shape)

#将强数据增强之后的image信息拿出来，转成RGB，再转成LAB形式
        img_rgb = []
        for i in range(len(img_metas)):
            img_after_norm = img[i]
            img_norm_cfg = img_metas[i]["img_norm_cfg"]
            img_before_norm = torch.tensor(color_transform(img_after_norm, **img_norm_cfg), device=img[0].device)
            img_before_norm = torch.clip(img_before_norm, min = 0, max = 255)
            img_rgb.append(img_before_norm)
        img_rgb = torch.stack(img_rgb, dim = 0).permute(0, 3, 1, 2)
        downsampled_images = F.avg_pool2d(
            img_rgb.float(), kernel_size=self.student.mask_assign_stride,
            stride=self.student.mask_assign_stride, padding=0
        )[:, [2, 1, 0]]

#计算color similarity--------------------------
        original_image_masks = [torch.ones_like(downsampled_images[0][0], dtype=torch.float32) for x in range(len(img_metas))]
        img_color_similarity = []
        for i in range(len(img_metas)):
            images_lab = color.rgb2lab(downsampled_images[i].byte().permute(1, 2, 0).cpu().numpy())
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = get_images_color_similarity(
                images_lab, original_image_masks[i],
                self.pairwise_size, self.pairwise_dilation
            )
            img_color_similarity.append(images_color_similarity.squeeze(0))
        img_color_similarity = torch.stack(img_color_similarity, dim = 0)


        for i, gt_mask in enumerate(pseudo_masks):
            #if i == 0:
            #    mask_tensor = torch.randn(0, gt_mask.height, gt_mask.width)
            #else:
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)
            #soft_mask_tensor = mask_tensor
            
#可视化---------------------------------------    不匹配 
            #如果是interval的倍数，才可视化图片
            if flag == 1:            
                image_vis = img[i].cpu().detach()
                mask_vis = mask_tensor.cpu().detach()
                mask_vis = mask_vis > 0.5
                label_vis = gt_labels[i].cpu().detach()

                if  mask_tensor.shape[0] > 0:
                    log_image_with_masks_without_box(
                        "pesudo_mask",
                        image_vis,
                        None,
                        mask_vis,
                        bbox_tag="pesudo_mask",
                        labels=label_vis,
                        class_names=self.CLASSES,
                        interval=1,
                        img_norm_cfg=img_metas[i]["img_norm_cfg"],
                    )
#二值化                
            mask_tensor = mask_tensor > 0.5
            mask_tensor = mask_tensor.float()   

#生成Box--------------------------------------                 
            bbox_list = [] 
            box_mask = None
            if mask_tensor.shape[0] != 0:
                for id in range(mask_tensor.shape[0]):
                    self.sum_psedudo_mask += 1
                    box_single_mask = None
                    mask_temp = mask_tensor[id]
                    coor = torch.nonzero(mask_temp)
                    if coor.shape[0] == 0:
                        self.sum_none_box += 1
                        xmin = 0
                        xmax = 0
                        ymin = 0
                        ymax = 0
                        box_single_mask = torch.zeros([assign_H, assign_W], device = img[0][0].device)
                    else:
                        xmin = torch.min(coor[:, 1]).item()
                        xmax = torch.max(coor[:, 1]).item()

                        ymin = torch.min(coor[:, 0]).item()
                        ymax = torch.max(coor[:, 0]).item()

                    bbox_list.append([xmin, ymin, xmax, ymax, teacher_info["det_ious"][i][id].item()])
                    if box_single_mask == None:
                        box_single_mask = torch.zeros([assign_H, assign_W], device = img[0][0].device)
                        box_single_mask[ymin // self.student.mask_assign_stride: ymax // self.student.mask_assign_stride + 1, xmin // self.student.mask_assign_stride: xmax // self.student.mask_assign_stride + 1] = 1

                    if box_mask == None:
                        box_mask = box_single_mask.unsqueeze(0)
                    else:
                        box_mask = torch.cat((box_mask, box_single_mask.unsqueeze(0)), 0)
            
            bbox_tensor = torch.tensor(bbox_list)

#可视化Box--------------------------------------            
            if flag == 1:
                image_vis = img[i].cpu().detach()
                label_vis = gt_labels[i].cpu().detach()
                bbox_vis = bbox_tensor
                if  mask_tensor.shape[0] > 0:
                    log_image_with_boxes(
                        "pesudo_box",
                        image_vis,
                        bbox_vis,
                        bbox_tag="pesudo_box",
                        labels=label_vis,
                        class_names=self.CLASSES,
                        interval=1,
                        img_norm_cfg=img_metas[i]["img_norm_cfg"],
                    )

#可视化Box的mask--------------------------------------  
            if flag == 1:            
                image_vis = F.interpolate(img[i].unsqueeze(0), (assign_H, assign_W), 
                    mode='bilinear', align_corners=False).squeeze(0).cpu().detach()
                mask_vis = box_mask
                label_vis = gt_labels[i].cpu().detach()

                if  mask_tensor.shape[0] > 0:
                    log_image_with_masks_without_box(
                        "pesudo_box_mask",
                        image_vis,
                        None,
                        mask_vis,
                        bbox_tag="pesudo_box_mask",
                        labels=label_vis,
                        class_names=self.CLASSES,
                        interval=1,
                        img_norm_cfg=img_metas[i]["img_norm_cfg"],
                    )
                
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
                '''
                gt_soft_masks_tensor.append(
                    soft_mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
                '''
                #gt_bbox_tensor.append([None])
                #gt_box_mask_tensor.append(mask_tensor.new_zeros((mask_tensor.size(0), assign_H, assign_W)))
                gt_box_mask_tensor.append(None)
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])
                '''
                gt_soft_masks_tensor.append(
                    F.interpolate(
                        soft_mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])
                '''
                #gt_bbox_tensor.append(bbox_tensor.float().to(img[0][0].device) // self.student.mask_assign_stride)
                gt_box_mask_tensor.append(box_mask)

#----------------------------------#
        #print("gt_masks_tensor[0].shape:", len(gt_masks_tensor), gt_masks_tensor[0].shape)  #2 torch.Size([14, 200, 336])
        #print("gt_labels.shape", len(gt_labels), gt_labels[0].shape)    #2 torch.Size([14])
        #print("gt_labels:", gt_labels)

#----记录全0的pseudo mask数量--------#
        '''
        flag_num = isVisualbyCount(50)
        if flag_num == 1:
            data_root = "/home/zsc/cc-project/ssl/SoftTeacher/work_dirs/ssl/mayue_test_ssl_ls_soft/record.txt"
            f = open(data_root, 'a+')
            if self.sum_psedudo_mask == 0:
                percent = self.sum_none_box / (self.sum_psedudo_mask + 1)
            else:
                percent = self.sum_none_box / self.sum_psedudo_mask
            per = str(percent)
            num_n = str(self.sum_none_box)
            num_s = str(self.sum_psedudo_mask)
            res = per + " " + num_n + " " + num_s + '\n'
            #print(res)
            f.writelines(res)
            self.sum_none_box = 0
            self.sum_psedudo_mask = 0
        '''

        gt_masks = gt_masks_tensor
        #gt_bboxes = gt_bbox_tensor
        #gt_soft_mask = gt_soft_masks_tensor
        gt_box_mask = gt_box_mask_tensor
        gt_scores =  teacher_info["det_scores"]
        gt_ious = teacher_info["det_ious"] 
        
        rpn_results = self.student.rpn_head.forward_train_unsup(x, img_metas, gt_masks, gt_box_mask, gt_labels, gt_scores, gt_ious, img_color_similarity=img_color_similarity)
        (rpn_losses, proposal_feats, x_feats, mask_preds, cls_scores) = rpn_results
        
        loss_rpn_seg = rpn_losses['loss_rpn_seg']
        rpn_losses['loss_rpn_seg'] = loss_rpn_seg * 0
        #rpn_losses['sum_none_box'] = self.sum_none_box
        #if self.sum_psedudo_mask == 0:
        #    rpn_losses['none_box_percent'] = self.sum_none_box / (self.sum_psedudo_mask + 1)
        #else:
        #    rpn_losses['none_box_percent'] = self.sum_none_box / self.sum_psedudo_mask
#------------------------------------#
        #print("mask_preds.shape:", mask_preds.shape)
        losses = self.student.roi_head.forward_train_unsup(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_labels,
            img_color_similarity,
            gt_box_mask,
            gt_scores,
            gt_ious,
            imgs_whwh=None)
        iou_loss = losses["s0_loss_iou"]
        losses["s0_loss_iou"] = iou_loss * 0
        iou_loss = losses["s1_loss_iou"]
        losses["s1_loss_iou"] = iou_loss * 0        
        iou_loss = losses["s2_loss_iou"]
        losses["s2_loss_iou"] = iou_loss * 0
        
        #对dice loss，mask loss， focal loss加权
        
        losses.update(rpn_losses)
        return losses        
   
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


#！ 修改
#!!!!!!!!!
    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        #不需要保存teacher的proposal
        
        
        #start2 = time.time()
        rpn_outs = self.teacher.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        #roi_outs = self.teacher.roi_head.simple_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        #num_roi_outs = len(roi_outs)
        #seg_results, label_results, score_results = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
#!这边有分支，可以选择带iou的，也可以选择不带iou的
#--------------------------------------------#
        #seg_results, label_results, score_results, _ = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        seg_results, label_results, score_results, iou_results = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        #end2 = time.time()
        #print("test_cost:", end2 - start2)
        
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):    #过滤阈值去除分值比较低的检测框和类别
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        
        if isinstance(self.train_cfg.pseudo_label_iou_thr, float):    #过滤阈值去除分值比较低的检测框和类别
            iou_thr = self.train_cfg.pseudo_label_iou_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")        
            
        # list(zip(*list)),将数组中的元组中的每一项取出,添加到一起,组成新的数组
        
#-两种过滤方式。带iou的和不带iou的
#----------------------------------#
        det_masks, det_labels, det_scores, det_ious = list(   
            zip(
                *[
                    filter_invalid_3(
                        mask=seg_result,
                        label=label_result,
                        score=score_result,
                        iou=iou_result,
                        thr=thr,
                        iou_thr=iou_thr
                    )
                    for seg_result, label_result, score_result, iou_result in zip(
                        seg_results, label_results, score_results, iou_results
                    )
                ]
            )
        )
        
#---------------------------------------------#        
        #print("det_masks[0].mask.shape:", det_masks[0].mask.shape)
        #print("det_labels[0].shape", det_labels[0].shape)
        #print("det_labels", det_labels)
        #print("det_masks", det_masks)
        #print("det_labels", det_labels)
        
        
        teacher_info["det_masks"] = det_masks
        teacher_info["det_labels"] = det_labels
        teacher_info["det_scores"] = det_scores
        teacher_info["det_ious"] = det_ious
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        #teacher_info["x_feats"] = x_feats
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
