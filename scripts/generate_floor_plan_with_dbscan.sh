#!/usr/bin/env bash

FILEPATH=$1   # 입력 PLY 파일 경로
if [ -z "$2" ]; then
  OUT_FILE="3d_floor_plan_files/$1.ply"  # 2번 arg가 없을 경우 기본 경로 설정
else
  OUT_FILE=$2  # 2번 arg가 있을 경우 해당 경로 사용
fi

python3 utils/generate_floorplan.py \
  --option dbscan \
  --filepath "$FILEPATH" \
  --room_name none \
  --out "$OUT_FILE"