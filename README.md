# Dashboard Variable Detection (YOLOv8 + OCR)

산업용 계기판(대시보드)에서 변수(PV, SV, 압력 등)를 감지하고, 필요 시 숫자 OCR까지 수행하는 프로젝트입니다.

## 프로젝트 구조

- `data/`
  - `samples/`: 테스트용 입력 샘플(이미지/XML)
  - `dataset/`, `cnn_dataset/`: 저장소에 포함하지 않는 재생성 대상 데이터
- `models/`: 학습된 가중치(`best.pt`, `cnn_digit_best.pth`)
- `tests/`
  - `test_ocr_image.py`: 이미지 감지 + OCR 결과 패널 표시
  - `test_video.py`: 비디오 감지 (옵션으로 OCR on/off)
- `tools/`
  - `extract_digits.py`: 샘플 이미지에서 숫자 crop 추출
  - `balance_classified_data.py`: 7-segment 합성 이미지 생성
  - `augment_cnn_data.py`: `classified` 기반 train/val 데이터셋 생성
  - `train_cnn.py`: CNN OCR 학습
- `prepare_dataset.py`: YOLO 데이터셋 생성/패키징

## 환경 설정

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 빠른 실행 가이드

### 1) YOLO 감지만 확인

```bash
python tests/test_video.py --conf 0.40
```

- 기본 비디오는 `data/samples/video_compat.mp4`를 우선 사용합니다.
- 해당 파일이 없으면 자동으로 `data/samples/video.mp4`를 시도합니다.

### 2) YOLO + OCR 함께 확인

```bash
python tests/test_video.py --ocr --conf 0.40
```

- OCR은 `models/cnn_digit_best.pth`가 있어야 동작합니다.
- 값이 비정상적으로 흔들리면 `--conf`를 올려 오탐지를 줄이세요.

### 3) 이미지 대시보드 방식 확인

```bash
python tests/test_ocr_image.py
```

## OCR 데이터 생성/학습 파이프라인

`data/cnn_dataset`은 저장소에서 제외되므로 아래 순서로 로컬 재생성합니다.

```bash
# 1) 샘플 이미지에서 숫자 crop 추출
python tools/extract_digits.py

# 2) 추출 결과를 수동 분류 (data/cnn_dataset/classified/{0..9,minus,dot,...})

# 3) 클래스 균형용 합성 이미지 추가 (선택)
python tools/balance_classified_data.py

# 4) train/val 증강 데이터 생성
python tools/augment_cnn_data.py

# 5) CNN 학습
python tools/train_cnn.py
```

## 데이터/용량 정책

- 대용량 산출물(`data/dataset`, `data/cnn_dataset`, `*.cache`, 실험 산출 디렉터리)은 `.gitignore`로 제외합니다.
- 장기 보관이 필요한 데이터는 별도 스토리지(artifact bucket, release asset 등)에 보관하세요.

## 참고

- 히스토리 경량화를 위해 과거 대용량 데이터(`data/cnn_dataset`, `data/images` 등)는 Git 히스토리에서 제거되었습니다.
