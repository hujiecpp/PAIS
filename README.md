# PAIS
Code release for paper "Pseudo-label Alignment for Semi-supervised Instance Segmentation" [ICCV 2023]

# Pseudo-label Alignment for Semi-supervised Instance Segmentation

## Usage

### Requirements
- `Ubuntu 16.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.9.0`
- `mmdetection=2.23.0`
- `mmcv=1.3.17`

### Data Preparation
- Download the COCO dataset and cityscapes dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco/
#     train2017/
#     val2017/

#     unlabeled2017/
#     annotations/
#  cityscapes/
#     leftImg8bit/
#     gtFine/
#     annotations/

```  
### Training
```shell script
# JOB_TYPE: 'baseline' or 'semi', decide which kind of job to run
# PERCENT_LABELED_DATA: 1, 5, 10. The ratio of labeled coco data in whole training dataset.
# GPU_NUM: number of gpus to run the job
bash tools/dist_train_partially_weight.sh <JOB_TYPE> ${FOLD} <PERCENT_LABELED_DATA> <GPU_NUM>
```
For example, we could run the following scripts to train our model on 10% labeled data with 4 GPUs:

```shell script
bash tools/dist_train_partially_weight.sh semi ${FOLD} 10 4
```

### Evaluation
```
bash tools/dist_test.sh <CONFIG_FILE_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval bbox --cfg-options model.test_cfg.rcnn.score_thr=<THR>
```

