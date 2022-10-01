#!/bin/bash

task_name=rim_lowvar
log_dir=logs_baseline/TN/$task_name/train
root_dir=checkpoint_baseline/$task_name
sample_dir=demo/mug_task_${task_name}.gzip
max_epoch=200
lr=0.0001

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

python3 ./baselines/baseline_train.py --plot-path=$log_dir/imgs/ --root-dir=$root_dir --sample-dir=$sample_dir --max-epoch=$max_epoch --lr=$lr --saveplot | tee $log_dir/output.txt