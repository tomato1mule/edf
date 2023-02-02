#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Using CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"





eval_name=eval_full
log_dir=logs/eval/$eval_name

eval_config_dir="config/eval_config/scalar_field_slow/$eval_name.yaml"
pick_agent_config_dir="config/agent_config/scalar_field/pick_agent.yaml"
place_agent_config_dir="config/agent_config/scalar_field/place_agent.yaml"
plot_path="logs/eval/$eval_name/plot/"
checkpoint_path_pick="checkpoint/train_pick/model_iter_600.pt"
checkpoint_path_place="checkpoint/train_place/model_iter_600.pt"


sleep 1
if [ ! -f $checkpoint_path_pick ] || [ ! -f $checkpoint_path_place  ]; then
    echo "ERROR: No checkpoint found. Please run train.sh first and then try again."
else
    echo "##### Begin Evaluation #####"
    echo Result images and outputs will be saved at $log_dir
    mkdir -p $log_dir
    PYTHONHASHSEED=0 python3 edf/eval.py --eval-config-dir=$eval_config_dir \
                                     --pick-agent-config-dir=$pick_agent_config_dir \
                                     --place-agent-config-dir=$place_agent_config_dir \
                                     --plot-path=$plot_path \
                                     --checkpoint-path-pick=$checkpoint_path_pick \
                                     --checkpoint-path-place=$checkpoint_path_place \
                                     --save-plot \
                                     | tee $log_dir/output.txt
    echo "##### Evaluation Done #####"
fi



eval_name=eval_upright
log_dir=logs/eval/$eval_name

eval_config_dir="config/eval_config/scalar_field_slow/$eval_name.yaml"
pick_agent_config_dir="config/agent_config/scalar_field/pick_agent.yaml"
place_agent_config_dir="config/agent_config/scalar_field/place_agent.yaml"
plot_path="logs/eval/$eval_name/plot/"
checkpoint_path_pick="checkpoint/train_pick/model_iter_600.pt"
checkpoint_path_place="checkpoint/train_place/model_iter_600.pt"

sleep 1
if [ ! -f $checkpoint_path_pick ] || [ ! -f $checkpoint_path_place  ]; then
    echo "ERROR: No checkpoint found. Please run train.sh first and then try again."
else
    echo "##### Begin Evaluation #####"
    echo Result images and outputs will be saved at $log_dir
    mkdir -p $log_dir
    PYTHONHASHSEED=0 python3 edf/eval.py --eval-config-dir=$eval_config_dir \
                                     --pick-agent-config-dir=$pick_agent_config_dir \
                                     --place-agent-config-dir=$place_agent_config_dir \
                                     --plot-path=$plot_path \
                                     --checkpoint-path-pick=$checkpoint_path_pick \
                                     --checkpoint-path-place=$checkpoint_path_place \
                                     --save-plot \
                                     | tee $log_dir/output.txt
    echo "##### Evaluation Done #####"
fi




eval_name=eval_pose
log_dir=logs/eval/$eval_name

eval_config_dir="config/eval_config/scalar_field_slow/$eval_name.yaml"
pick_agent_config_dir="config/agent_config/scalar_field/pick_agent.yaml"
place_agent_config_dir="config/agent_config/scalar_field/place_agent.yaml"
plot_path="logs/eval/$eval_name/plot/"
checkpoint_path_pick="checkpoint/train_pick/model_iter_600.pt"
checkpoint_path_place="checkpoint/train_place/model_iter_600.pt"

sleep 1
if [ ! -f $checkpoint_path_pick ] || [ ! -f $checkpoint_path_place  ]; then
    echo "ERROR: No checkpoint found. Please run train.sh first and then try again."
else
    echo "##### Begin Evaluation #####"
    echo Result images and outputs will be saved at $log_dir
    mkdir -p $log_dir
    PYTHONHASHSEED=0 python3 edf/eval.py --eval-config-dir=$eval_config_dir \
                                     --pick-agent-config-dir=$pick_agent_config_dir \
                                     --place-agent-config-dir=$place_agent_config_dir \
                                     --plot-path=$plot_path \
                                     --checkpoint-path-pick=$checkpoint_path_pick \
                                     --checkpoint-path-place=$checkpoint_path_place \
                                     --save-plot \
                                     | tee $log_dir/output.txt
    echo "##### Evaluation Done #####"
fi




eval_name=eval_instance
log_dir=logs/eval/$eval_name

eval_config_dir="config/eval_config/scalar_field_slow/$eval_name.yaml"
pick_agent_config_dir="config/agent_config/scalar_field/pick_agent.yaml"
place_agent_config_dir="config/agent_config/scalar_field/place_agent.yaml"
plot_path="logs/eval/$eval_name/plot/"
checkpoint_path_pick="checkpoint/train_pick/model_iter_600.pt"
checkpoint_path_place="checkpoint/train_place/model_iter_600.pt"

sleep 1
if [ ! -f $checkpoint_path_pick ] || [ ! -f $checkpoint_path_place  ]; then
    echo "ERROR: No checkpoint found. Please run train.sh first and then try again."
else
    echo "##### Begin Evaluation #####"
    echo Result images and outputs will be saved at $log_dir
    mkdir -p $log_dir
    PYTHONHASHSEED=0 python3 edf/eval.py --eval-config-dir=$eval_config_dir \
                                     --pick-agent-config-dir=$pick_agent_config_dir \
                                     --place-agent-config-dir=$place_agent_config_dir \
                                     --plot-path=$plot_path \
                                     --checkpoint-path-pick=$checkpoint_path_pick \
                                     --checkpoint-path-place=$checkpoint_path_place \
                                     --save-plot \
                                     | tee $log_dir/output.txt
    echo "##### Evaluation Done #####"
fi




eval_name=eval_distractor
log_dir=logs/eval/$eval_name

eval_config_dir="config/eval_config/scalar_field_slow/$eval_name.yaml"
pick_agent_config_dir="config/agent_config/scalar_field/pick_agent.yaml"
place_agent_config_dir="config/agent_config/scalar_field/place_agent.yaml"
plot_path="logs/eval/$eval_name/plot/"
checkpoint_path_pick="checkpoint/train_pick/model_iter_600.pt"
checkpoint_path_place="checkpoint/train_place/model_iter_600.pt"

sleep 1
if [ ! -f $checkpoint_path_pick ] || [ ! -f $checkpoint_path_place  ]; then
    echo "ERROR: No checkpoint found. Please run train.sh first and then try again."
else
    echo "##### Begin Evaluation #####"
    echo Result images and outputs will be saved at $log_dir
    mkdir -p $log_dir
    PYTHONHASHSEED=0 python3 edf/eval.py --eval-config-dir=$eval_config_dir \
                                     --pick-agent-config-dir=$pick_agent_config_dir \
                                     --place-agent-config-dir=$place_agent_config_dir \
                                     --plot-path=$plot_path \
                                     --checkpoint-path-pick=$checkpoint_path_pick \
                                     --checkpoint-path-place=$checkpoint_path_place \
                                     --save-plot \
                                     | tee $log_dir/output.txt
    echo "##### Evaluation Done #####"
fi




