#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Error: 파일 이름을 입력해주세요"
    echo "Usage: sh scripts/prepare_data.sh <파일_이름>"
    exit 1
fi

FILE_NAME=$1  # 예: scaniverse-model 62

python3 dataset/scannetv2/prepare_data_inst.py --file_name "$FILE_NAME"