MODEL_PATH="/mnt/longcontext/models/siyuan/llama3/QwQ-32B-Yarn"
MODEL_TEMPLATE_TYPE="qwq"
MODEL_TASK="think_test"
# get the base name of the model path
MODEL_NAME=$(basename $MODEL_PATH)
MODEL_DIR=$(dirname $MODEL_PATH)
TEST_LEN="131072"
export MODEL_FRAMEWORK_EXTRA="vllm"

cd ..
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
pip install causal-conv1d>=1.2.0
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
pip install huggingface_hub==0.24 --no-deps
echo "Installation complete."
cd ../scripts/data/think_test/json/
python download_paulgraham_essay.py && echo "Downloaded Paul Graham essay dataset"
bash download_qa_dataset.sh && echo "Downloaded QA dataset"
cd ../../../../
echo $$pwd
sleep 5
ls -l
sleep 5
cd scripts
python download_nltk.py
which python
pip install -U tiktoken
pip list
export only_last_logits=1
# bash run.sh QwQ-32B-Yarn think_test /mnt/longcontext/models/siyuan/llama3 qwq 4 "131072"
bash run.sh $MODEL_NAME $MODEL_TASK $MODEL_DIR $MODEL_TEMPLATE_TYPE 4 $TEST_LEN