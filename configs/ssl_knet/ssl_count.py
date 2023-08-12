_base_ = "base.py"

semi_wrapper = dict(
    type="SslKnet",
    test_cfg=dict(inference_on="teacher"),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
    val=dict(
        type="Coco_iou",
        ann_file='data/coco/annotations/semi_supervised/instances_train2017.1@5000.json',
        img_prefix='data/coco/train2017/',
        pipeline=test_pipeline),
    test=dict(
        type="Coco_iou",
        ann_file='data/coco/annotations/semi_supervised/instances_train2017.1@5000.json',
        img_prefix='data/coco/train2017/',
        pipeline=test_pipeline)
)

fold = 1
percent = 1

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
#work_dir="work_dirs/ssl/3"
work_dir="work_dirs/ssl/baseline_with_iou_1"
resume_from = "work_dirs/ssl/baseline_with_iou_1/latest.pth"
#resume_from = None
#runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=12)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
