#!/bin/bash

# Detect the CUDA version
cuda_version=$(nvcc --version | grep -oP "release \K\d+\.\d+")

echo "Detected CUDA version: $cuda_version"

# Install PyTorch
if [[ "$cuda_version" == "11.8" ]]; then
    echo "Installing PyTorch 2.1.0 for CUDA 11.8..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$cuda_version" == "12.2" ]]; then
    echo "Installing the latest PyTorch version for CUDA 12.2..."
    pip install torch torchvision torchaudio
else
    echo "Installing the latest PyTorch version for CUDA ${cuda_version}... though it may not be compatible."
    pip install torch torchvision torchaudio
fi

pip install Cython
pip install packaging

echo "Installing the required Python packages..."

pip install -r requirements.txt --no-deps

echo "Installing the TransformerEngine package..."

# pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

echo "Installing the FlashAttention package..."

pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary --no-deps

pip install numpy==1.23.5 --no-deps
pip install huggingface_hub==0.23.2 --no-deps
pip install nltk --no-deps
pip install regex --no-deps

python -c "import nltk; nltk.download('punkt_tab')"

