import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import build_assigner, build_sampler, mask_matrix_nms
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead
from .mask_pseudo_sampler import MaskPseudoSampler
from mmdet.core.mask.structures import BitmapMasks
from mmcv.runner import force_fp32, auto_fp16


@HEADS.register_module()
class KernelIterHead(BaseRoIHead):

    def __init__(self,
                 num_stages=6,
                 recursive=False,
                 assign_stages=5,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 merge_cls_scores=False,
                 do_panoptic=False,
                 post_assign=False,
                 hard_target=False,
                 num_proposals=100,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 thing_label_in_seg=0,
                 mask_head=dict(
                     type='KernelUpdateHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_out_stride=4,
                 train_cfg=None,
                 test_cfg=None,
                 
                 **kwargs):
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        self.merge_cls_scores = merge_cls_scores
        self.recursive = recursive
        self.post_assign = post_assign
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.assign_stages = assign_stages
        self.do_panoptic = do_panoptic
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_thing_classes + num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.num_proposals = num_proposals
        self.fp16_enabled = False
        super(KernelIterHead, self).__init__(
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            
            **kwargs)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(
                    self.mask_sampler[stage], MaskPseudoSampler), \
                    'Sparse Mask only support `MaskPseudoSampler`'

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.mask_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.mask_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))
    
    def init_weights(self):
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()
    

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if self.recursive:
            for i in range(self.num_stages):
                self.mask_head[i] = self.mask_head[0]

    def _mask_forward(self, stage, x, object_feats, mask_preds, img_metas):
        mask_head = self.mask_head[stage]
        cls_score, iou_scores, mask_preds, object_feats = mask_head(
            x, object_feats, mask_preds, img_metas=img_metas)
        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
        else:
            scaled_mask_preds = mask_preds
        mask_results = dict(
            cls_score=cls_score,
            iou_scores=iou_scores,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats)

        return mask_results

    #@auto_fp16(apply_to=('x', 'proposal_feats', 'mask_preds', 'cls_score', 'gt_masks')) 
    @force_fp32(apply_to=('x', 'proposal_feat', 'mask_preds', 'cls_score', 'gt_masks'))
    def forward_train(self,
                      x,
                      proposal_feats,
                      mask_preds,
                      cls_score,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_scores=None,
                      gt_ious=None,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_bboxes=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None):

        num_imgs = len(img_metas)
        if self.mask_head[0].mask_upsample_stride > 1:
            prev_mask_preds = F.interpolate(
                mask_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
        else:
            prev_mask_preds = mask_preds.detach()

        if cls_score is not None:
            prev_cls_score = cls_score.detach()
        else:
            prev_cls_score = [None] * num_imgs

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks

        object_feats = proposal_feats
        all_stage_loss = {}
        all_stage_mask_results = []
        assign_results = []
        for stage in range(self.num_stages):
            
#-------------------------------#
            #!对数据进行判断，如果是无标签数据则使用rce_loss
            flag = 1
        
            if img_metas[0]["tag"] == "sup":
                #print("sup"+"="*100)
                #print("flag", flag)
                flag = 1
            else:
                flag = 0             

            
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score']
            object_feats = mask_results['object_feats']
            iou_scores = mask_results['iou_scores']

            if self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()

            sampling_results = []
            if stage < self.assign_stages:
                assign_results = []
            for i in range(num_imgs):
                if stage < self.assign_stages:
                    mask_for_assign = prev_mask_preds[i][:self.num_proposals]
                    if prev_cls_score[i] is not None:
                        cls_for_assign = prev_cls_score[
                            i][:self.num_proposals, :self.num_thing_classes]
                    else:
                        cls_for_assign = None
                    if flag == 0:
                        assign_result = self.mask_assigner[stage].assign(
                            mask_for_assign, cls_for_assign, gt_masks[i],
                            gt_labels[i], img_metas[i], gt_scores[i], gt_ious[i])
                    else:
                        assign_result = self.mask_assigner[stage].assign(
                            mask_for_assign, cls_for_assign, gt_masks[i],
                            gt_labels[i], img_metas[i])
                    assign_results.append(assign_result)
                sampling_result = self.mask_sampler[stage].sample(
                    assign_results[i], scaled_mask_preds[i], gt_masks[i])
                sampling_results.append(sampling_result)
#----------------------------#            
            num_batch = len(gt_masks)
            #print("num_batch:", num_batch)
            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                gt_masks,
                gt_labels,
                self.train_cfg[stage],
                True,
                num_batch,
                gt_scores,
                gt_ious,
                flag,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls)


            single_stage_loss = self.mask_head[stage].loss(
                object_feats,
                cls_score,
                iou_scores,
                scaled_mask_preds,
                flag,
                *mask_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f's{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]

            if not self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()

        return all_stage_loss

#!待修改，增加iou预测    
    def simple_test(self,
                    x,
                    proposal_feats,
                    mask_preds,
                    cls_score,
                    img_metas,
                    imgs_whwh=None,
                    rescale=False):

        # Decode initial proposals
        num_imgs = len(img_metas)
        # num_proposals = proposal_feats.size(1)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            iou_scores = mask_results['iou_scores']

        num_classes = self.mask_head[-1].num_classes
        results = []

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

#-------------------------#            
        #if self.mask_head[-1].loss_iou.use_sigmoid:
        iou_scores = iou_scores.sigmoid()
        
        #暂未支持全景的iou
        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(cls_score[img_id],
                                                  scaled_mask_preds[img_id],
                                                  self.test_cfg,
                                                  img_metas[img_id])
                results.append(single_result)
        else:
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                
                iou_scores_per_img = iou_scores[img_id]
                
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg.max_per_img, sorted=True)
                mask_indices = topk_indices // num_classes
                labels_per_img = topk_indices % num_classes
                masks_per_img = scaled_mask_preds[img_id][mask_indices]
                iou_per_img = iou_scores_per_img.flatten(0)[mask_indices]
#用来生成指定分类阈值的伪mask                
                
                """
                valid_0 = scores_per_img > 0.3
                valid_1 = iou_per_img > 0.8
                valid = valid_0 * valid_1
                
                scores_per_img = scores_per_img[valid]
                labels_per_img = labels_per_img[valid]
                masks_per_img = masks_per_img[valid]
                #print("scores_per_img", type(scores_per_img), scores_per_img)
                #print("masks_per_img", type(masks_per_img), masks_per_img)
                """
                #print("iou_per_img", iou_per_img)
                #print("scores_per_img", scores_per_img)
                
            
                single_result = self.mask_head[-1].get_seg_masks(
                    masks_per_img, labels_per_img, scores_per_img,
                    self.test_cfg, img_metas[img_id])
                results.append(single_result)
        return results

#待修改，增加iou预测
    def teacher_test(self,
                    x,
                    proposal_feats,
                    mask_preds,
                    cls_score,
                    img_metas,
                    imgs_whwh=None,
                    rescale=False):
        num_imgs = len(img_metas)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            iou_scores = mask_results['iou_scores']
        num_classes = self.mask_head[-1].num_classes
        results = []
        
        score_list = []
        label_list = []
        iou_list = []
        
        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()  #沿着最后一维即类别维度进行归一化
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]        
        #print("cls_score:", cls_score)  #
        #print("cls_score.shape:", cls_score.shape) #torch.Size([1, 100, 80])
#----------------------------------------#
        #if self.mask_head[-1].loss_iou.use_sigmoid:
        iou_scores = iou_scores.sigmoid()
        #不经过这个分支
        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(cls_score[img_id],
                                                  scaled_mask_preds[img_id],
                                                  self.test_cfg,
                                                  img_metas[img_id])
                results.append(single_result)
        else:
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                #cls_score_list.append(cls_score_per_img)

                iou_scores_per_img = iou_scores[img_id]
                
                #print("cls_score_per_img.flatten(0, 1)", cls_score_per_img.flatten(0, 1).shape)  #torch.Size([8000])
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg.max_per_img, sorted=True)        #max_per_img = 100, flatten从起始维度到目标维度推平， topk,默认维度1，即[]中取top, 返回值和索引
                #
                #print("scores_per_img:", scores_per_img) 
                #print("scores_per_img.shape:", scores_per_img.shape) #torch.Size([100])
                #print("topk_indices:", topk_indices) #
                
                mask_indices = topk_indices // num_classes
                
                #print("mask_indices:", mask_indices)  #会有同一个kernel产生的结果，但是是不同的类别
                
                labels_per_img = topk_indices % num_classes
                
                #print("labels_per_img:", labels_per_img) 
                #print("scores_per_img:", scores_per_img)
                
                masks_per_img = scaled_mask_preds[img_id][mask_indices]        
               
                iou_per_img = iou_scores_per_img.flatten(0)[mask_indices]
                
                #print("masks_per_img:", masks_per_img)
                #print(masks_per_img.shape)  #torch.Size([100, 200, 304])

                seg_masks = self.rescale_masks(masks_per_img, img_metas[img_id])
                #seg_masks = seg_masks > self.test_cfg.mask_thr         # 二值化
                
                
                #print("scores_per_img:", scores_per_img.shape)
                score_list.append(scores_per_img)
                label_list.append(labels_per_img)
                iou_list.append(iou_per_img)
                
                
                _, h, w = seg_masks.shape
                bmask = seg_masks.cpu().numpy()
                bitmask = BitmapMasks(bmask, h, w)
                results.append(bitmask)
        #print("len(result)", len(result), len(score_list), len(label_list))                 
        return results, label_list, score_list, iou_list
        #return results, label_list, score_list

    

    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        '''
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks
        '''
        masks_per_img = masks_per_img.squeeze(0)
        return masks_per_img
    
    

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('SparseMask does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_feats, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_mask_results = []
        num_imgs = len(img_metas)
        num_proposals = proposal_feats.size(1)
        C, H, W = x.shape[-3:]
        mask_preds = proposal_feats.bmm(x.view(num_imgs, C, -1)).view(
            num_imgs, num_proposals, H, W)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
        return all_stage_mask_results

    def get_panoptic(self, cls_scores, mask_preds, test_cfg, img_meta):
        # resize mask predictions back
        scores = cls_scores[:self.num_proposals][:, :self.num_thing_classes]
        thing_scores, thing_labels = scores.max(dim=1)
        stuff_scores = cls_scores[
            self.num_proposals:][:, self.num_thing_classes:].diag()
        stuff_labels = torch.arange(
            0, self.num_stuff_classes) + self.num_thing_classes
        stuff_labels = stuff_labels.to(thing_labels.device)

        total_masks = self.mask_head[-1].rescale_masks(mask_preds, img_meta)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)

        panoptic_result = self.merge_stuff_thing(total_masks, total_labels,
                                                 total_scores,
                                                 test_cfg.merge_stuff_thing)
        return dict(pan_results=panoptic_result)

    def merge_stuff_thing(self,
                          total_masks,
                          total_labels,
                          total_scores,
                          merge_cfg=None):

        H, W = total_masks.shape[-2:]
        panoptic_seg = total_masks.new_full((H, W),
                                            self.num_classes,
                                            dtype=torch.long)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_thing_classes
            if isthing and total_scores[k] < merge_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < merge_cfg.overlap_thr:
                    continue

                panoptic_seg[mask] = total_labels[k] \
                    + current_segment_id * INSTANCE_OFFSET
                current_segment_id += 1

        return panoptic_seg.cpu().numpy()
