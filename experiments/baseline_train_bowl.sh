#!/bin/bash

task_name=bowl_task_rim
log_dir=logs_baseline/TN/$task_name/train
root_dir=checkpoint_baseline/$task_name
sample_dir=demo/${task_name}.gzip
max_epoch=200
lr=0.0001
save_freq=200
pick_grasp=top
place_grasp=top

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

python3 ./baselines/baseline_train.py --plot-path=$log_dir/imgs/ --root-dir=$root_dir --sample-dir=$sample_dir --max-epoch=$max_epoch --lr=$lr --saveplot --save-freq=$save_freq --pick-grasp=$pick_grasp --place-grasp=$place_grasp | tee $log_dir/output.txt