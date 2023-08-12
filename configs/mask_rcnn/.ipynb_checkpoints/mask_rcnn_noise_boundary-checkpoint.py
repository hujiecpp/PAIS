mmdet_base = "../../thirdparty/mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/models/mask_rcnn_r50_fpn.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py"
]

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


work_dir = "work_dirs/mask_rcnn/noise_boundary"
resume_from = "work_dirs/iter_60000.pth"