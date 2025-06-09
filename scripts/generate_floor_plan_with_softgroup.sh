#!/usr/bin/env bash

ROOM_NAME=$1  # 예: scaniverse-model 62
if [ -z "$2" ]; then
  OUT_FILE="3d_floor_plan_files/$1.ply"  # 2번 arg가 없을 경우 기본 경로 설정
else
  OUT_FILE=$2  # 2번 arg가 있을 경우 해당 경로 사용
fi

python3 utils/generate_floor_plan.py \
  --option softgroup \
  --filepath none \
  --room_name "$ROOM_NAME" \
  --out "$OUT_FILE"