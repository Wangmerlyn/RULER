# This script is used to test the RULER model with the specified parameters.
# bash ruler_local_refine.sh | tee ruler.log
export MODEL_FRAMEWORK_EXTRA="vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SUPER_FORCE_138K=true
export VLLM_FORCE_128K=true
# use anwser prefix = 1 is for original
export RULER_USE_ANSWER_PREFIX=0
export RULER_TEMPERATURE=0.6
export RULER_TOKENS_TO_GENERATE=8192
export RULER_EVAL_POSTPROCESS=r1
INSTALL_ENV=1
TEST_MODEL_PATH="/mnt/longcontext/models/siyuan/llama3/DeepSeek-R1-Distill-Llama-8B"
TEMPALTE="llama_distill_r1"
TEST_LENS="4096,8192,16384,32768,65536,131072"
echo "Test settings:"
echo "======================================"
echo "TEST_MODEL_PATH: $TEST_MODEL_PATH"
echo "TEMPALTE: $TEMPALTE"
echo "TEST_LENS: $TEST_LENS"
echo "INSTALL_ENV: $INSTALL_ENV"
echo "MODEL_FRAMEWORK_EXTRA: $MODEL_FRAMEWORK_EXTRA"
echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
echo "RULER_USE_ANSWER_PREFIX: $RULER_USE_ANSWER_PREFIX"
echo "RULER_TEMPERATURE: $RULER_TEMPERATURE"
echo "RULER_TOKENS_TO_GENERATE: $RULER_TOKENS_TO_GENERATE"
echo "RULER_EVAL_POSTPROCESS: $RULER_EVAL_POSTPROCESS"
echo "======================================"


# get the name of the model
TEST_MODEL_NAME="$(basename $TEST_MODEL_PATH)"
TEST_MODEL_FOLDER="$(dirname $TEST_MODEL_PATH)"
cd RULER
# if install_env==1,
if [ $INSTALL_ENV -eq 1 ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda create --name ruler python=3.10 -y
    source /opt/conda/etc/profile.d/conda.sh
    conda activate ruler
    which python
    echo $$PATH
    echo "which pip"
    which pip
    git clone https://github.com/Wangmerlyn/RULER
    cd RULER
    git checkout think_reg
    cd docker
    echo "custom installation"
    pip install vllm -U
    pip install Cython
    pip install packaging
    echo "Installing the required Python packages..."
    pip install nemo_toolkit[all] --user
    pip install numba==0.60.0
    pip install flask
    pip install flask_restful
    pip install sshtunnel_requests
    pip install tritonclient[all]
    pip install wonderwords
    pip install openai
    pip install tiktoken
    pip install tenacity
    pip install transformers
    pip install flash-attn
    pip install accelerate
    pip install huggingface_hub
    pip install causal-conv1d
    pip install mamba-ssm
    pip install html2text
    pip install bs4
    pip install pandas
    pip install google-generativeai
    echo "reinstalling torch"
    pip install nltk==3.8.1
    pip install regex
    pip install pyyaml
    pip install tqdm
    pip install hydra-core
    pip install omegaconf
    pip install pytorch-lightning
    echo "Installation complete."
    cd ../scripts/data/think_test/json/
    python download_paulgraham_essay.py && echo "Downloaded Paul Graham essay dataset"
    bash download_qa_dataset.sh && echo "Downloaded QA dataset"
    cd ../../../../
    echo $$pwd
    sleep 5
    ls -l
    sleep 5
    python download_nltk.py
    which python
    pip install -U tiktoken
    pip list
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate ruler
cd scripts
export only_last_logits=1
# bash run.sh DeepSeek-R1-Distill-Llama-8B think_test /mnt/longcontext/models/siyuan/llama3 llama_distill_r1 4 "4096,8192,16384,32768,65536,131072"
bash run.sh $TEST_MODEL_NAME think_test $TEST_MODEL_FOLDER $TEMPALTE 4 $TEST_LENS
# /mnt/longcontext/models/siyuan/llama3/DeepSeek-R1-Distill-Llama-8B