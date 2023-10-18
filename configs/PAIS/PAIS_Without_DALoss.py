_base_ = "base.py"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
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
            sample_ratio=[1, 3],
        )
    ),
)

fold = 1
percent = 10

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl_percent/10/baseline_0.65"
#work_dir="work_dirs/mayue_iccv23/ema_knet_0.5_0.3_1-2"

#resume_from = work_dir="work_dirs/mayue_iccv23/ema_knet_0.5_0.3_1-2/iter_160000.pth"
resume_from = "work_dirs/ssl/baseline_3/iter_120000.pth"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
