_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]
num_stages = 3
num_proposals = 100
conv_kernel_size = 1

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    ), 
    neck=dict(
        init_cfg=dict(
            type="Normal", layer='Conv2d', std=0.01
        ),
    ), 
    rpn_head=dict(
        localization_fpn=dict(
            init_cfg=dict(
                type="Normal", layer='Conv2d', std=0.01,
            ),   
        ), 
        init_cfg=[dict(
            type="Normal", layer='Conv2d', std=0.01, 
            override=dict(type='Normal', name='conv_seg', std=0.01, bias_prob=0.01)
        ), 
        dict(
            type="Normal", layer='Conv2d', std=0.01, 
            override=dict(type='Normal', name='init_kernels', std=1, mean=0)
        )],
    ), 
    roi_head=dict(
        type='KernelIterHead',
        mask_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),                
                init_cfg=dict(
                    type="Xavier", layer=['Conv2d', 'Linear'],
                    override=[dict(type='Constant', name='fc_cls', val=1, bias_prob=0.01), 
                             dict(type='Normal', name='fc_mask', mean=0, std=0.01),
                             ]
                    
                    )
                ) for _ in range(num_stages)
            ]
    ), 
)

fold = 1
percent = 10

work_dir="work_dirs/knet/2"

resume_from = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/semi_supervised/instances_train2017.${fold}@${percent}.json',
        img_prefix=data_root + 'train2017/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))

checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=20)
evaluation = dict(metric=['segm'], interval=8000)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[120000, 160000])

