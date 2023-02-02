#!/bin/bash

use_gui=0




sleep 1
echo "Begin Evaluation for Arbitrary Pose, Instance and Distractors"

task_name=bowl_task_rim
eval_name=eval_full

log_dir=logs_baseline/TN/$task_name/eval/$eval_name
root_dir=checkpoint_baseline/$task_name

init_seed=100
end_seed=200
mug_pose=arbitrary
mug_type=cups
task_type=bowl_task

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

if [ $use_gui -gt 0 ]; then
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --use-gui --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type --use-support --distractor | tee $log_dir/output.txt
else
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type --use-support --distractor | tee $log_dir/output.txt
fi




sleep 1
echo "Begin Evaluation for Default Configurations"

task_name=bowl_task_rim
eval_name=eval_upright

log_dir=logs_baseline/TN/$task_name/eval/$eval_name
root_dir=checkpoint_baseline/$task_name

init_seed=100
end_seed=200
mug_pose=upright
mug_type=default
task_type=bowl_task

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

if [ $use_gui -gt 0 ]; then
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --use-gui --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type | tee $log_dir/output.txt
else
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type | tee $log_dir/output.txt
fi




sleep 1
echo "Begin Evaluation for Unseen Poses"

task_name=bowl_task_rim
eval_name=eval_lying

log_dir=logs_baseline/TN/$task_name/eval/$eval_name
root_dir=checkpoint_baseline/$task_name

init_seed=100
end_seed=200
mug_pose=lying
mug_type=default
task_type=bowl_task

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

if [ $use_gui -gt 0 ]; then
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --use-gui --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type --use-support | tee $log_dir/output.txt
else
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type --use-support | tee $log_dir/output.txt
fi









sleep 1
echo "Begin Evaluation for Unseen Instances"

task_name=bowl_task_rim
eval_name=eval_instance

log_dir=logs_baseline/TN/$task_name/eval/$eval_name
root_dir=checkpoint_baseline/$task_name

init_seed=100
end_seed=200
mug_pose=upright
mug_type=cups
task_type=bowl_task

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

if [ $use_gui -gt 0 ]; then
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --use-gui --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type | tee $log_dir/output.txt
else
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type | tee $log_dir/output.txt
fi









sleep 1
echo "Begin Evaluation for Distractors"

task_name=bowl_task_rim
eval_name=eval_distractor

log_dir=logs_baseline/TN/$task_name/eval/$eval_name
root_dir=checkpoint_baseline/$task_name

init_seed=100
end_seed=200
mug_pose=upright
mug_type=default
task_type=bowl_task

if [ ! -f "$log_dir" ]; then
    mkdir -p $log_dir
fi

if [ $use_gui -gt 0 ]; then
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --use-gui --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type --distractor| tee $log_dir/output.txt
else
    python3 ./baselines/baseline_eval.py --checkpoint-iter=1000 --plot-path=$log_dir/imgs/ --root-dir=$root_dir --save-plot --task-type=$task_type --init-seed=$init_seed --end-seed=$end_seed --mug-pose=$mug_pose --mug-type=$mug_type --distractor| tee $log_dir/output.txt
fi






















