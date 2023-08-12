_base_ = "base.py"

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
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
            sample_ratio=[1, 2],
        )
    ),
)

fold = 1
percent = 1

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl_baseline/1_1-2"

#resume_from = "/hdd1/cc/ssl/ema/baseline_1-2/iter_72000.pth"
resume_from = None

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
