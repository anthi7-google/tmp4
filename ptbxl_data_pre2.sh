#!/bin/sh

set -e

# paths
ROOT_DIR="data_pt"
SRC_DIR="$ROOT_DIR/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
OUT_DIR="$ROOT_DIR/processed_100hz"
SAMPLING_RATE="100" 
# 100 or 500

#pip install numpy pandas wfdb --break-system-packages

# output dir
mkdir -p "$OUT_DIR"

# run conversion
python3 preprocess_ptbxl.py \
    --src_dir "$SRC_DIR" \
    --save_dir "$OUT_DIR" \
    --sampling_rate "$SAMPLING_RATE"