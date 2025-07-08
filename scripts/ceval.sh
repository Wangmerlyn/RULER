#!/bin/bash

BENCHMARK="think_test"
model_name_list=(
    "qwen25_fix_no_entro_2node_16k_2k_math_filtered_dis_numchain64_mathqa_512bsz_20ksamples_grpo_subem_end-step43"
    "qwen25_fix_no_entro_2node_16k_2k_math_filtered_dis_numchain64_mathqa_512bsz_20ksamples_grpo_subem_end-step43-Yarn"
)

export RULER_EVAL_POSTPROCESS="r1"
pred_suffix="pred"
lens=(4096 8192 16384 32768 65536 131072)

for model_name in "${model_name_list[@]}"; do
  (
    pred_prefix="/mnt/longcontext/models/siyuan/RULER_think/${model_name}/think_fix"
    for len in "${lens[@]}"; do
      PRED_DIR="${pred_prefix}/${len}/${pred_suffix}"
      echo "Evaluating model: ${model_name} | Length: ${len}"
      python eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
    done
  ) &
done

wait  # Wait for all background processes to finish