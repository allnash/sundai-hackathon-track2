#!/bin/bash
set -e
cd "$(dirname "$0")"

ENV_NAME="cuda-challenge"

echo "Creating conda environment '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" python=3.11

echo ""
echo "Installing dependencies..."
conda run -n "$ENV_NAME" pip install torch ninja numpy huggingface_hub

echo ""
echo "Setup complete. Run ./benchmark.sh to test your solution."
