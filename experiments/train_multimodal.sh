#!/bin/bash

seed=0
task_name=mug_task
pick_type=mixed
task_name=${task_name}_${pick_type}
demo_dir=./demo/$task_name.gzip

if [ ! -f "$demo_dir" ]; then
    echo "Generating oracle demo at: $demo_dir"
    mkdir -p logs/demo/$task_name
    PYTHONHASHSEED=$seed python3 generate_demo.py --use-gui --file-name="$task_name.gzip" --pick-type="$pick_type" | tee logs/demo/$task_name/output.txt
    echo "Demo generation done"
else
    echo "Using existing demo file: $demo_dir"
fi


export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Using CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

sleep 5
echo "##### Begin Place Training with Seed $seed #####"
mkdir -p logs/train/place_agent
PYTHONHASHSEED=$seed python3 edf/train_place.py --visualize --save-plot --save-checkpoint --sample-dir="$demo_dir" --seed=$seed | tee logs/train/place_agent/output.txt
echo "##### Place Training Done #####"

sleep 5
echo "##### Begin Pick Training with Seed $seed #####"
mkdir -p logs/train/pick_agent
PYTHONHASHSEED=$seed python3 edf/train_pick.py --visualize --save-plot --save-checkpoint --sample-dir="$demo_dir" --seed=$seed | tee logs/train/pick_agent/output.txt
echo "##### Pick Training Done #####"

