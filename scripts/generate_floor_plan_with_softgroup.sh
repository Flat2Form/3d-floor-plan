#!/usr/bin/env bash

# 사용법 출력 함수
usage() {
  echo "사용법: $0 <room_name> [output_file]"
  echo "  room_name: 입력 PLY 파일 이름"
  echo "  output_file: (선택) 출력 파일 경로. 미지정시 3d_floor_plan_files/<room_name>.ply 사용"
  exit 1
}

# 1번 인자 체크 
if [ -z "$1" ]; then
  usage
fi

ROOM_NAME=$1   # 입력 PLY 파일 이름

if [ -z "$2" ]; then
  OUT_FILE="3d_floor_plan_files/$ROOM_NAME.ply"  # 2번 arg가 없을 경우 기본 경로 설정
else
  OUT_FILE=$2  # 2번 arg가 있을 경우 해당 경로 사용
fi

echo "입력 파일: $ROOM_NAME"


if [ -f "dataset/scannetv2/data/$ROOM_NAME.ply" ]; then
  sh scripts/prepare_data.sh;
fi
if [ -f "dataset/scannetv2/test/"$ROOM_NAME"_inst_nostuff.pth" ]; then
  sh scripts/run_inference.sh;
fi
python3 utils/generate_floor_plan.py \
  --option softgroup \
  --filepath none \
  --room_name "$ROOM_NAME" \
  --out "$OUT_FILE";