# 사용 예시
# sh scripts/visualize_ply.sh scaniverse_model_62 scaniverse_model_62_pred.ply

#!/usr/bin/env bash

ROOM_NAME=$1  # 예: scaniverse-model 62
OUT_FILE=$2   # 예: scaniverse_model_62_pred.ply

python3 tools/bounding_box.py \
  --prediction_path results/ \
  --room_name "$ROOM_NAME" \
  --task bounding_box \
  --out "$OUT_FILE"
