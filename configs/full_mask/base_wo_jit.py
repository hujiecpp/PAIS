mmdet_base = "../../thirdparty/mmdetection/configs/_base_"

#！ 修改
#!!!!!!!!!
_base_ = [
    f"{mmdet_base}/models/mask_rcnn_r50_fpn.py",
    f"{mmdet_base}/datasets/coco_instance.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
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
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_masks'],
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

strong_pipeline = [
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
                type="ShuffledSequential",
                transforms=[
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
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_masks'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

#！ 修改collect
#!!!!!!!!!
weak_pipeline = [
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
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_masks'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

#！ 修改PseudoSamples, 并且去类中查看是否支持with_mask=True，PseudoSamples类有些疑惑，可以参考LoadAnnotations的源码来理解
#!!!!!!!!!
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True, with_mask=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

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
    samples_per_gpu=None,
    workers_per_gpu=None,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="CocoDataset",
            ann_file=None,
            img_prefix=None,
            pipeline=train_pipeline,
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file=None,
            img_prefix=None,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=False,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

semi_wrapper = dict(
    type="SoftTeacher_wo_jit",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=1.5,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]
evaluation = dict(type="SubModulesDistEvalHook", interval=7500, metric=['bbox', 'segm'])

optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60000, 80000])

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=7500, max_keep_ckpts=20)

fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ])