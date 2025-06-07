#!/usr/bin/env bash

FILE_NAME=$1  # 예: scaniverse-model 62

# cd dataset/scannetv2
python3 dataset/scannetv2/prepare_data_inst.py --file_name "$FILE_NAME"
# cd ../..