_base_ = "base.py"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
            sample_ratio=[2, 2],
        )
    ),
)

fold = 1
percent = 10

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl_mask_rcnn/v5"

resume_from = None
#resume_from = "work_dirs/ssl_mask_rcnn/v4/iter_18750.pth"

lr_config = dict(step=[60000, 80000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=90000)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
