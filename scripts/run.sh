#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name> $2 <model_directory> $3 <template> $4 <batchsize> $5 <seq_lengths> $6"
    # exit 1
fi


# Root Directories
# check if GPUS is set, if not, set it to 4
if [ -z "$GPUS" ]; then
    echo "GPUS is not set, defaulting to 4"
    GPUS=4
fi
ROOT_DIR="/mnt/longcontext/models/siyuan/RULER_think" # the path that stores generated task samples and model predictions.
MODEL_DIR=$3 # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=$5  # increase to improve GPU utilization

echo $MODEL_DIR


# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

if [ -n "$4" ]; then
    MODEL_TEMPLATE_TYPE="$4"
else
    MODEL_TEMPLATE_TYPE="Phi3.1"
fi

echo "MODEL_TEMPLATE:$MODEL_TEMPLATE_TYPE"

echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

echo "BATCHSIZE:$BATCH_SIZE"
SEQ_LENGTHS=$6
IFS=',' read -r -a SEQ_LENGTHS <<< "$6"
echo "SEQ_LENGTHS:${SEQ_LENGTHS[@]}"


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# check if MODEL_FRAMEWORK_EXTRA is set
if [ -n "${MODEL_FRAMEWORK_EXTRA}" ]; then
    MODEL_FRAMEWORK="${MODEL_FRAMEWORK_EXTRA}"
fi

echo "MODEL_FRAMEWORK_EXTRA: $MODEL_FRAMEWORK_EXTRA"
echo "MODEL_FRAMEWORK: $MODEL_FRAMEWORK"


# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    # Check if the VLLM_FORCE_128K environment variable is set to 'true'
    # check env var SUPER_FORCE_129K
    if [ "$SUPER_FORCE_138K" == "true" ]; then
        # If true, add --max-model-len 141072 to the command
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        echo "SUPER_FORCE_138K is set to true, using --max-model-len 141072"
        # check if "yarn" in MODEL_PATH.lower(), if so, use --hf-overrides
        model_lower="${MODEL_PATH,,}"
        if [[ "$model_lower" == *"yarn"* || ( "$model_lower" == *"deepseek-r1"* && "$model_lower" == *"qwen3"* ) ]]; then
            echo "Matched yarn or deepseek distill+qwen3"
            # If the model is a yarn model, use the yarn rope scaling
            python pred/serve_vllm.py \
                --model=${MODEL_PATH} \
                --tensor-parallel-size=${GPUS} \
                --dtype bfloat16 \
                --disable-custom-all-reduce \
                --seed 0 \
                --max-model-len 141072 \
                --hf-overrides  '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 35268}}' \
                &
        else
            # If the model is not a yarn model, use the default rope scaling
            python pred/serve_vllm.py \
                --model=${MODEL_PATH} \
                --tensor-parallel-size=${GPUS} \
                --dtype bfloat16 \
                --disable-custom-all-reduce \
                --seed 0 \
                --max-model-len 141072 \
                --hf-overrides "{\"max_position_embeddings\": 141072}" \
                &
        fi
    else

        if [ "$VLLM_FORCE_128K" == "true" ]; then
            # If true, add --max-model-len 131072 to the command
            export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
            echo "VLLM_FORCE_128K is set to true, using --max-model-len 131072"
            python pred/serve_vllm.py \
                --model=${MODEL_PATH} \
                --tensor-parallel-size=${GPUS} \
                --dtype bfloat16 \
                --disable-custom-all-reduce \
                --seed 0 \
                --max-model-len 131072 \
                --hf-overrides "{\"max_position_embeddings\": 131072}" \
                &
        else
            # If VLLM_FORCE_128K is not 'true', run the command without --max-model-len
            python pred/serve_vllm.py \
                --model=${MODEL_PATH} \
                --tensor-parallel-size=${GPUS} \
                --dtype bfloat16 \
                --disable-custom-all-reduce \
                --seed 0 \
                &
        fi
    fi

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        &

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer \
        &
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs

fi


# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    if [ -n "$override_save_name" ]; then
        RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${override_save_name}/${MAX_SEQ_LENGTH}"
    else
        RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    fi
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
        
        start_time=$(date +%s)
        python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done
    
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
