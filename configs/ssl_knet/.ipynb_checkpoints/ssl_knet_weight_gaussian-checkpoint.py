_base_ = "base_weight_gaussian.py"

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
#work_dir="work_dirs/ema_weight_2_4"
work_dir="work_dirs/test_gaussian_before_erzhi_3"

resume_from = "work_dirs/ema_weight_2_3/iter_80000.pth"
#resume_from = None

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
