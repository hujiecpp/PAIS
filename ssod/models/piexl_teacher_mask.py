import torch
import numpy as np
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n, log_image_with_masks, log_image_with_masks_without_box, isVisualbyCount

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid, filter_invalid_bbox_iou, filter_invalid_mask_iou
from mmdet.core.mask.structures import BitmapMasks
import torch.nn.functional as F


@DETECTORS.register_module()
class PiexlTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(PiexlTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

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
            sup_loss = self.student.forward_train(**data_groups["sup"])
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
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

#!修改
#-----------
    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )
        
        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],   # list[torch.tensor]  3, 0
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        pseudo_bbox_ious = teacher_info["det_bbox_ious"]
        pseudo_mask_ious = teacher_info["det_mask_ious"]
#！由于两个通道的数据增强不一致，需要将teacher的mask变换到student上，mask的变换----------------------#
        """
        if isinstance(teacher_info["det_masks"][0], np.ndarray):
            pseudo_masks = teacher_info["det_masks"]
            pseudo_masks = [torch.from_numpy(pseudo_masks[i]).to(M[0].device).to(M[0].dtype) for i in range(0, len(pseudo_masks))]
        else:
            pseudo_masks = self._transform_mask(
                teacher_info["det_masks"],   #list(list)    3, 80, 0
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
            pseudo_masks = [pseudo_masks[i].masks for i in range(0, len(pseudo_masks))]
        """
        #print("+"*100)
        #print("teacher_info_det_masks", teacher_info["det_masks"])
        
        pseudo_masks = self._transform_mask(
            teacher_info["det_masks"],   #list(bitmapmasks)    3, 
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        
        interval = 1000
        flag = isVisualbyCount(interval)
        
        gt_masks = []
        for i, gt_mask in enumerate(pseudo_masks):
            pad_H, pad_W = student_info["img_metas"][i]['batch_input_shape']
            mask_tensor = gt_mask.to_tensor(torch.float, pseudo_labels[i].device)
            
            '''
            
            if flag == 1:
                image_vis = student_info["img"][i].cpu().detach()
                mask_vis = mask_tensor.cpu().detach()
                bbox_vis = pseudo_bboxes[i][:, :4].cpu().detach()
                label_vis = pseudo_labels[i].cpu().detach()
                
                print("mask_vis", mask_vis.shape)
                print("img_norm_cfg", student_info["img_metas"][i]["img_norm_cfg"])
                print("image_vis", image_vis.shape)
                
                if bbox_vis.shape[0] > 0:
                    log_image_with_masks_without_box(
                        "before_pad",
                        image_vis,
                        bbox_vis,
                        mask_vis,
                        bbox_tag="before_pad",
                        labels=label_vis,
                        class_names=self.CLASSES,
                        interval=1,
                        img_norm_cfg=student_info["img_metas"][i]["img_norm_cfg"],
                    )
            '''
            #不padding的话，尺度对不上
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)  
            mask_tensor = mask_tensor > 0.5
            
            if flag == 1:
                image_vis = student_info["img"][i].cpu().detach()
                mask_vis = mask_tensor.cpu().detach()
                bbox_vis = pseudo_bboxes[i][:, :4].cpu().detach()
                label_vis = pseudo_labels[i].cpu().detach()
                '''
                print("mask_vis", mask_vis.shape)
                #print("img_norm_cfg", student_info["img_metas"][i]["img_norm_cfg"])
                print("image_vis", image_vis.shape)
                print("batch_input_shape", student_info["img_metas"][i]['batch_input_shape'])
                print("img_shape", student_info["img_metas"][i]['img_shape'])
                '''
                if bbox_vis.shape[0] > 0:
                    log_image_with_masks_without_box(
                        "after_pad",
                        image_vis,
                        bbox_vis,
                        mask_vis,
                        bbox_tag="after_pad",
                        labels=label_vis,
                        class_names=self.CLASSES,
                        interval=1,
                        img_norm_cfg=student_info["img_metas"][i]["img_norm_cfg"],
                    )
                
            #mask_tensor = mask_tensor > 0.5
            mask_tensor = mask_tensor.float().cpu().numpy()
            _, h, w = mask_tensor.shape
            bitmask = BitmapMasks(mask_tensor, h, w)
            gt_masks.append(bitmask)
        pseudo_masks = gt_masks
#---------------------------#
        
        
        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

#!0.9过滤的伪标签进行mask监督            
        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                pseudo_masks,
                pseudo_bbox_ious,
                pseudo_mask_ious,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        '''
#!这边是重新算回归loss，bbox更少
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                pseudo_masks,
                student_info=student_info,
            )
        )
        '''

        """
        loss.update(
            self.unsup_rcnn_mask_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                pseudo_masks,
                student_info=student_info,
            )
        )
        """
#---------------------------#
    
        return loss
    

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,   #利用 rpn_pseudo_threshold=0.9 阈值过滤伪框
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=1000,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        pseudo_masks,
        pseudo_bbox_ious,
        pseudo_mask_ious,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        losses = {}
        gt_bboxes, gt_labels, gt_ious, gt_masks, gt_mask_ious = multi_apply(
            filter_invalid_mask_iou,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            pseudo_bbox_ious,
            pseudo_masks,
            pseudo_mask_ious,
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        '''
#看看得到的mask的尺度是什么样的------------------------------ 不对齐的,且不是28*28，而是原图大小左右的尺度#
        print("伪mask的尺度"+"-"*100)
        print("student_info_img", student_info["img"][0].shape)   #torch.Size([3, 1024, 1344])
        print("gt_masks[0].masks.shape", gt_masks[0].masks.shape) # (1, 1000, 1333)
        print("", )
        '''
        
        
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        
        
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        #命名不规范，这边是tuple
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        
        #softteacher的操作
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach() #命名不规范
            
            
        
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            bbox_results["iou_score"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        
        #print("bbox_targets", bbox_targets.shape)
        
        #print("loss_cls", loss["loss_cls"].shape)
        #print("loss_bbox", loss["loss_bbox"]) #(n, 4)
        
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        loss["loss_iou"]  = loss["loss_iou"].sum() * 0
        losses.update(loss)
        
        
        pos_rois = bbox2roi([res.pos_bboxes[:, :4] for res in sampling_results])
        mask_results = self.student.roi_head._mask_forward(feat, pos_rois)
        mask_targets = self.student.roi_head.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.student.train_cfg.rcnn)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        
        
        #print("pos_labels", pos_labels.shape)
        
        
        
        num_imgs = len(img_metas)
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_mask_iou_list = []
        for i in range(num_imgs):
            if pos_assigned_gt_inds_list[i].numel() == 0:
                pos_mask_iou = pos_assigned_gt_inds_list[i].new_zeros((0, 1))
            else:
                pos_mask_iou = gt_mask_ious[i][pos_assigned_gt_inds_list[i]]
            pos_mask_iou_list.append(pos_mask_iou)
        pos_mask_iou =  torch.cat(pos_mask_iou_list)
        
        #print("pos_mask_iou", pos_mask_iou.shape)
        
        loss_mask = self.student.roi_head.mask_head.loss_unlabel(mask_results['mask_pred'],
                                        mask_targets, pos_labels, mask_results['mask_iou_score'], pos_mask_iou)
        
        #print("loss_mask", loss_mask["loss_mask"].shape)
        
        loss_mask["loss_mask"] = loss_mask["loss_mask"].sum() / max(
            mask_targets.size()[0], 1.0
        )
        #loss_mask["loss_mask"] = loss_mask["loss_mask"] / 2.0
        loss_mask["loss_mask"] = loss_mask["loss_mask"] / 2.0
        
        loss_mask["loss_mask_iou"] = loss_mask["loss_mask_iou"] * 0
        losses.update(loss_mask)
        
        
        '''
        interval = 1000
        flag = isVisualbyCount(interval)
        if flag == 1:
    #只可视化batch size中的第一张图片的gt        
            pad_H, pad_W = img_metas[0]['batch_input_shape']
            mask_tensor = gt_masks[0].to_tensor(torch.float, gt_labels[0].device)
            if gt_masks[0].width != pad_W or gt_masks[0].height != pad_H:
                pad_wh = (0, pad_W - gt_masks[0].width, 0, pad_H - gt_masks[0].height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)        

    #测试mask的变换是否有问题        --------------尺度对上了
            
            #print("pad之后的mask"+"-"*100)
            #print("student_info_img", student_info["img"][0].shape)
            #print("mask_tensor.shape", mask_tensor.shape)
            
            image_vis = student_info["img"][0].cpu().detach()
            mask_vis = mask_tensor.cpu().detach()
            bbox_vis = gt_bboxes[0].cpu().detach()
            label_vis = gt_labels[0].cpu().detach()
            
            if len(gt_bboxes[0]) > 0:
                log_image_with_masks_without_box(
                    "rcnn_cls_mask",
                    image_vis,
                    bbox_vis,
                    mask_vis,
                    bbox_tag="rcnn_cls_mask",
                    labels=label_vis,
                    class_names=self.CLASSES,
                    interval=1000,
                    img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
                )
       
        '''
        return losses


#！**************
#!加上mask的loss, 后续还需要看半监督语义分割的论文来进行修改。*********
    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        pseudo_masks,
        student_info=None,
        **kwargs,
    ):
        """
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        
        print("=============================")
        print("pseudo_bboxes:", pseudo_bboxes)
        print(pseudo_bboxes[0].shape)
        print("=============================")
        print("pseudo_masks:", pseudo_masks)
        #print(pseudo_masks[0].shape)        
        """
        
        gt_bboxes, gt_labels, gt_masks = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            pseudo_masks,
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        
        """
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        """
        loss = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, gt_masks=gt_masks, unlabel_mask_flag=0, **kwargs
        )
        loss_bbox = loss["loss_bbox"]
        #loss_mask = loss["loss_mask"]
        
        
        #print("", )
        
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=1000,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        #return {"loss_bbox": loss_bbox}
        return {"loss_bbox": loss_bbox}


    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

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

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

#！ 修改
#!!!!!!!!!
    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
#-------------------------------------------------------#            
            """
            #print("rpn_out:", rpn_out)      [[tensor([[[[ ]]]])， ], ]
            print("len(rpn_out):", len(rpn_out))    #2
            print("rpn_out[0]", rpn_out[0], len(rpn_out[0]))  #5 ，多尺度的特征图
            print("rpn_out[0][0].shape", rpn_out[0][0].shape)  #torch.Size([3, 3, 176, 264])  176, 264会变化
            #get_bboxes在BaseDenseHead中
            
            """
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            
            #print("len(proposal_list):", len(proposal_list))        #3
            #print("type(proposal_list[0]):", type(proposal_list[0]))        #tensor
            #print("proposal_list:", proposal_list)      #[tensor, , ] tensor(n, 5)
            
            #print("proposal_list[0].shape:", proposal_list[0].shape)   #torch.Size([1000, 5])
        
        
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list, proposal_iou_list = self.teacher.roi_head.simple_test_bboxes_iou(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

#-----------------------------------------------#        
        #print("len(proposal_list):", len(proposal_list))   #3
        #print("proposal_list[0]", proposal_list[0])        #
        #print("proposal_list[0].shape", proposal_list[0].shape)  #torch.Size([100, 5])
        

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        
        proposal_iou_list = [p.to(feat[0].device) for p in proposal_iou_list]
        
        
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):    #过滤阈值去除分值比较低的检测框和类别
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
            
        # list(zip(*list)),将数组中的元组中的每一项取出,添加到一起,组成新的数组
        proposal_list, proposal_label_list, proposal_iou_list, _ = list(   
            zip(
                *[
                    filter_invalid_bbox_iou(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        iou=proposal_iou,
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label, proposal_iou in zip(
                        proposal_list, proposal_label_list, proposal_iou_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        
        
#！具体修改————增加mask分支-----------------------------------#
        #print(".........................")
        #print(len(proposal_list))
        
        simple_test_det_bboxes = list(proposal_list)
        simple_test_det_labels = list(proposal_label_list)
        
        det_masks, det_mask_ious = self.teacher.roi_head.tea_test_mask_iou(
            feat, img_metas, simple_test_det_bboxes, simple_test_det_labels, rescale=False
        )
        teacher_info["det_masks"] = det_masks 
        teacher_info["det_mask_ious"] = det_mask_ious 
#!---------------------------------------------------------#

    
        #计算分支不确定，box jettering
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        det_bbox_ious = proposal_iou_list
        
        '''
        print("ssl"+"-"*100)
        print("det_mask_ious", det_mask_ious[0].shape)
        print("det_bbox_ious", det_bbox_ious[0].shape)
        print("det_bboxes", det_bboxes[0].shape)
        print("det_labels", det_labels[0].shape)
        print("det_masks", det_masks[0].masks.shape)
        '''
        teacher_info["det_bbox_ious"] = det_bbox_ious
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

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
