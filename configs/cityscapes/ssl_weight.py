_base_ = "base_weight.py"
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root +
                'annotations/instancesonly_filtered_gtFine_train.1@30.json',
                img_prefix=data_root + 'leftImg8bit/train/',
                #pipeline=train_pipeline,
            ),
        ),
        unsup=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root +
                'annotations/instancesonly_filtered_gtFine_train.1@30-unlabeled.json',
                img_prefix=data_root + 'leftImg8bit/train/',
                #pipeline=unsup_pipeline,
                #filter_empty_gt=False,
            ),
            #pipeline=unsup_pipeline,
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 3],
        )
    ),
)

fold = 1
percent = 10


semi_wrapper = dict(
    type="PiexlTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.8,
        cls_pseudo_threshold=0.8,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=1.0,
    ),
    test_cfg=dict(inference_on="student"),
)

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl_mask_rcnn_cityscapes/30/1_3_0.5_0.8_0.8_1.0_adamw"

#resume_from = "work_dirs/ssl_mask_rcnn_cityscapes/20/1_4_0.5_0.9_0.9_1.0/iter_22000.pth"
#resume_from = "work_dirs/ssl_mask_rcnn/1/2_2_iou_180k/latest.pth"

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))

lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[2650*9],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=3000*9)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
