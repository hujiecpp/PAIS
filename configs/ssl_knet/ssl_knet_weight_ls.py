_base_ = "base_weight_ls.py"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            #ann_file="data/coco/annotations/instances_train2017.1@95-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

fold = 1
percent = 1

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
#work_dir="work_dirs/ssl/mayue_test_ssl"
work_dir="work_dirs/ssl/mayue_test_ssl_bs_11/"

#resume_from = "work_dirs/ssl/ema_weight_0.35_0.5_1-2/iter_112000.pth"
#resume_from = "work_dirs/ssl/mayue_test_ssl_ls_1_220k/latest.pth"
#load_from = "work_dirs/ssl/ema_iou_0.35_0.5_0.3/iter_72000.pth"
#load_from = "work_dirs/ssl/mayue_test_ssl_ls_3/iter_24000.pth"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
