mmdet_base = "../../thirdparty/mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/models/faster_rcnn_r50_fpn.py",
    f"{mmdet_base}/datasets/coco_detection.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]

model = dict(
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=3,
        )
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100))
    
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data_poc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1080, 1920), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 1920),
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


classes = ('BT', 'CK', 'BR')
#classes = ('work', 'OE', 'S', 'O_DIRTY', 'BL', 'B', 'O_RUST', 'D', 'H', 'OE_DOOR', 'OE_BEAM', 'OE_COLUMN', 'M')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-1,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train_v2.1@20-train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'annotations/instances_train_v2.1@20-train.json',
            img_prefix=data_root + 'train/',
            pipeline=train_pipeline
        )),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline)
)

evaluation = dict(interval=1, metric='bbox')

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

fp16 = dict(loss_scale="dynamic")

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir = "work_dirs/poc_v5/"