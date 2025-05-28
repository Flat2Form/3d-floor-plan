# 사용 예시
# sh scripts/visualize_ply.sh scaniverse_model_62 scaniverse_model_62_pred.ply

#!/usr/bin/env bash

ROOM_NAME=$1  # 예: scaniverse-model 62
OUT_FILE=$2   # 예: scaniverse_model_62_pred.ply

python tools/visualization.py \
  --prediction_path results/ \
  --room_name "$ROOM_NAME" \
  --task instance_pred \
  --out "$OUT_FILE"

