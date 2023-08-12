mmdet_base = "../../thirdparty/mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/models/mask_rcnn_r50_fpn.py",
    f"{mmdet_base}/datasets/coco_instance.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]

model = dict(
    roi_head=dict(
        type='StandardRoIHead_iou',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead_iou',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            #reg_decoded_bbox=True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead_iou',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    
)
custom_imports = dict(
    imports=[
        'ssod.mask_rcnn_iou.cls_bbox_iou',
        'ssod.mask_rcnn_iou.bbox_head',
        'ssod.mask_rcnn_iou.standard_roi_head_iou',
        'ssod.mask_rcnn_iou.fcn_mask_head_iou',
    ],
    allow_failed_imports=False)


#img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type='OneOf',
        transforms = [
            dict(type='GaussianBlur', sigma_limit=[0,3], p=1.0),
            dict(type='Blur', blur_limit=[2,7], p=1.0),
            dict(type='Sharpen', alpha=(0.,1.), lightness=(0.75,1.5), p=1.0),
            dict(type='GaussNoise', var_limit=(0.0, 0.05*255), p=1.0),
            dict(type='InvertImg', p=0.05),
            dict(type='MultiplicativeNoise', multiplier=(0.5, 1.5)),
            dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1.0),
            ], p=7/8
        )
    ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1333, 400), (1333, 1200)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu', transforms=albu_train_transforms, 
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'gt_masks': 'masks',
            },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CutOut', n_holes=(1,5), cutout_shape=[(0,0), (33,20), (66,40), (132,80), (198,120), (264,160)] ),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/semi_supervised/instances_train2017.${fold}@${percent}.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

fold = 1
percent = 10

evaluation = dict(metric=['bbox', 'segm'])

lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60000, 80000])
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=7500)
evaluation = dict(metric=['bbox', 'segm'], interval=7500)


work_dir = "work_dirs/mask_rcnn/noise_boundary_iou"
resume_from = None