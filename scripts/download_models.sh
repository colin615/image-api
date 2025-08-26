#!/usr/bin/env bash
set -e

MODEL_DIR=${MODEL_DIR:-/app/models}
mkdir -p "$MODEL_DIR"

echo "Downloading sample model artifacts to $MODEL_DIR (won't overwrite existing files)"

# SuperGlue repo (contains example weights & code; helpful later)
if [ ! -d "$MODEL_DIR/SuperGluePretrainedNetwork" ]; then
  git clone --depth 1 https://github.com/magicleap/SuperGluePretrainedNetwork.git "$MODEL_DIR/SuperGluePretrainedNetwork" || true
fi

# Real-ESRGAN sample weights (optional)
mkdir -p "$MODEL_DIR/realesrgan/weights"
if [ ! -f "$MODEL_DIR/realesrgan/weights/RealESRGAN_x4plus.pth" ]; then
  wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P "$MODEL_DIR/realesrgan/weights" || true
fi

echo "Done. Models are under $MODEL_DIR. You can edit this script to add other model downloads."
