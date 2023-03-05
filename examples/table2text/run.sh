#!/bin/bash

data_dir=${1}
task_mode=${2}
model_name_or_path=${3:-"gpt2"} # One of distilgpt2, gpt2, gpt2-medium, gpt2-large
num_train_epochs=${4:-"12"}
noise_multiplier=${5:-"1.4"}
per_device_train_batch_size=${6:-"16"}
batch_size=${7:-"1024"}
gpu_id=${8:-"1"}
gradient_accumulation_steps=`expr ${batch_size} / ${per_device_train_batch_size}`
output_dir=output/${task_mode}_${model_name_or_path}_${num_train_epochs}_${noise_multiplier}_${batch_size}

if [[ ${task_mode} == "e2e" ]]; then
  data_dir="${data_dir}/data/e2e_data"
  target_delta=8e-6
  learning_rate=2e-4
  max_seq_len=100
else
  if [[ ${task_mode} == "dart" ]]; then
    target_delta=1e-5
    data_dir="${data_dir}/data/dart"
    learning_rate=5e-5  # Lower learning rate for stability in large models.
    max_seq_len=120
  else
    echo "Unknown task: ${task_mode}"
    exit 1
  fi
fi

# Arguments in the last two lines are the most important.
CUDA_VISIBLE_DEVICES=${gpu_id} python -m table2text.run_language_modeling \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --task_mode ${task_mode} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps 10000000 --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps -1 \
  --seed 0 \
  --eval_steps 100 --eval_epochs 2 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "no" --evaluate_during_training "yes" --per_device_eval_batch_size 100 \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm 1 --target_delta ${target_delta} \
  --learning_rate ${learning_rate} --lr_decay "no" --num_train_epochs ${num_train_epochs} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --non_private no \
  --clipping_mode default \
  --cache_dir .cache \
  --noise_multiplier ${noise_multiplier}
