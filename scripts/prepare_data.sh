#!/usr/bin/env bash
# dataset/scannetv2/data 폴더의 모든 .ply 파일에 대해 처리
for file in dataset/scannetv2/data/*.ply; do
    if [ -f "$file" ]; then
        # 파일 이름에서 .ply 확장자 제거
        FILE_NAME=$(basename "$file" .ply)
        python3 dataset/scannetv2/prepare_data_inst.py --file_name "$FILE_NAME"
        echo "처리 완료: $FILE_NAME"
    fi
done
