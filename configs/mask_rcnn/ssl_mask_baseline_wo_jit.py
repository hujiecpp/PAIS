_base_ = "base_wo_jit.py"

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
            sample_ratio=[1, 3],
        )
    ),
)

fold = 1
percent = 10

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl_mask_rcnn/1/baseline_wo_jit_1_3"

#resume_from = "work_dirs/ssl_mask_rcnn/soft_v1/iter_60000.pth"
resume_from = None

lr_config = dict(step=[60000*2, 80000*2])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=90000*2)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
