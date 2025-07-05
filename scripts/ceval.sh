BENCHMARK="think_test"
model_name="qwen25_fix_no_entro_2node_16k_2k_math_filtered_dis_numchain64_mathqa_512bsz_20ksamples_grpo_subem_end-step43"
pred_prefix="/mnt/longcontext/models/siyuan/RULER_think/$model_name/think_fix"
export RULER_EVAL_POSTPROCESS="r1"
pred_suffix="pred"
lens=(4096 8192 16384 32768 65536 131072)
for len in "${lens[@]}"; do
    PRED_DIR="${pred_prefix}/${len}/${pred_suffix}"
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done