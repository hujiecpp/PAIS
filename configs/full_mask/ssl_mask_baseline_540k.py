_base_ = "base_wo_jit.py"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/instances_unlabeled2017.json",
            img_prefix="data/coco/unlabeled2017/",
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
work_dir="work_dirs/ssl_mask_rcnn/100/"

#resume_from = "work_dirs/ssl_mask_rcnn/100/iter_360000.pth"
resume_from = "work_dirs/ssl_mask_rcnn/100/iter_360000.pth"
#resume_from = None

optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)

lr_config = dict(step=[60000*6, 80000*6])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=90000*6)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
