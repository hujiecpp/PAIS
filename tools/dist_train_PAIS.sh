#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29105}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/baseline/.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/ssl_knet/ssl_knet_weight.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5} >> out/PAIS.out &
fi