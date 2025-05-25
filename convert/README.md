# Convert

3D 스캔 데이터(ply)를 SoftGroup 모델의 입력 포맷으로 변환하는 스크립트들이 있는 폴더입니다.

## 폴더 구조
`convert/
├── raw/ # 원본 데이터 폴더
│ ├── pending/ # 변환 대기 중인 ply 파일들
│ └── done/ # 변환이 완료된 ply 파일들
├── convert_pending_to_pth.py # ply 파일을 pth 파일로 변환하는 스크립트
└── README.md # 이 문서`

## 사용 방법

1. 변환하고자 하는 ply 파일을 `raw/pending/` 폴더에 복사합니다.

2. 변환 스크립트를 실행합니다:
```bash
python convert_pending_to_pth.py
```

3. 변환이 완료된 파일은 자동으로 `raw/done/` 폴더로 이동됩니다.

4. 변환된 pth 파일은 `dataset/scannetv2/test/` 폴더에 저장됩니다.