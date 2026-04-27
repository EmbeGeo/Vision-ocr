# Vision OCR

ESP32-CAM에서 MJPEG 스트림을 받아 YOLOv8으로 계기판 영역을 감지하고, EasyOCR로 수치를 판독해 외부 서버로 전송하는 라즈베리파이 기반 서비스입니다.

## 시스템 흐름

```
ESP32-CAM (MJPEG 스트림)
        │
        ▼
VideoPullClient   ← HTTP /stream 수신, 최신 프레임 유지
        │
        ▼
YOLOv8 (best.pt)  ← 계기판 영역 감지 및 클래스(변수명) 분류
        │
        ▼
EasyOcrRecognizer ← 크롭 이미지에서 숫자 판독 (변수별 전처리 튜닝)
        │
        ▼
ValueStabilizer   ← 최근 N회 결과 중 최빈값으로 떨림 억제
        │
        ▼
DataSender        ← 이상치 검증 후 주기적으로 JSON HTTP POST
```

## 파일 구조

```
vision-ocr/
├── main.py                        # 프로덕션 진입점 (전체 파이프라인 통합)
│
├── ocr/
│   ├── easyocr_recognizer.py      # EasyOCR 기반 숫자 인식기 (변수별 임계값 튜닝)
│   ├── video_client.py            # ESP32-CAM MJPEG 스트림 수신 클라이언트
│   ├── data_sender.py             # OCR 결과 주기 전송 + 이상치 검증
│   ├── stabilizer.py              # 최빈값 기반 값 안정화 (ValueStabilizer)
│   ├── compat.py                  # PyTorch 2.6+ torch.load 호환성 패치
│   ├── __init__.py
│   └── legacy/
│       └── recognizer.py          # 구버전 CNN 인식기 (비사용)
│
├── models/
│   └── best.pt                    # YOLOv8 학습된 모델
│
├── tests/
│   ├── test_ocr_esp32.py          # ESP32 라이브 스트림 OCR 테스트 (독립 실행 가능)
│   ├── test_ocr_image_send.py     # 이미지 1장 OCR 후 JSON 전송 테스트
│   ├── test_ocr_image.py          # 이미지 단순 OCR 확인
│   ├── test_ocr_video.py          # 동영상 파일 OCR 확인
│   └── eval_detection.py          # YOLO 감지 성능 평가
│
├── data/
│   └── samples/
│       ├── Full1.jpeg             # 테스트용 계기판 이미지
│       ├── Full1.xml              # 라벨 (Pascal VOC)
│       ├── Full1box.xml           # 바운딩박스 라벨
│       ├── Data2.png              # 추가 샘플
│       └── test_video.mp4         # 테스트용 영상
│
├── scratch/
│   ├── check_idx.py               # 데이터셋 인덱스 확인 유틸
│   └── test_split.py              # train/val 분할 테스트
│
├── colab_train.ipynb              # Google Colab YOLOv8 학습 노트북
├── prepare_dataset.py             # 데이터셋 전처리 스크립트
└── requirements.txt               # 의존성 패키지
```

## 주요 모듈

| 모듈 | 역할 |
|------|------|
| `EasyOcrRecognizer` | 변수명별 HSV 밝기 임계값과 EasyOCR allowlist(`0-9.-`)로 숫자 판독 |
| `VideoPullClient` | ESP32-CAM `/stream` 엔드포인트를 백그라운드 스레드로 수신, 최신 프레임 유지 |
| `ValueStabilizer` | 최근 5회 판독값 중 최빈값 반환 (OCR 노이즈 억제) |
| `DataSender` | 최대값(`500`) 및 급변(±10) 검증 후 주기적으로 JSON HTTP POST |

## 설치

```bash
pip install -r requirements.txt
pip install easyocr
```

> `requirements.txt`에 `paddleocr`/`paddlepaddle`이 포함되어 있으나, 현재 인식기는 EasyOCR을 사용합니다.

## 실행

### 프로덕션 (main.py)

```bash
# 기본 실행 (화면 출력 포함)
python main.py --ip 192.168.0.100

# 결과를 외부 서버로 전송
python main.py --ip 192.168.0.100 --send-to 192.168.0.200

# Headless 모드 (화면 없이 전송만)
python main.py --ip 192.168.0.100 --send-to 192.168.0.200 --no-show

# 전체 옵션
python main.py \
  --ip 192.168.0.100        # ESP32-CAM IP (필수)
  --send-to 192.168.0.200   # 수신 서버 IP
  --target-port 8000        # 수신 서버 포트 (기본: 8000)
  --target-path /ocr        # 수신 엔드포인트 (기본: /ocr)
  --interval 10             # 전송 주기 초 (기본: 10)
  --model models/best.pt    # YOLO 모델 경로
  --conf 0.5                # YOLO 감지 임계값
  --skip 5                  # OCR 판독 간격 (N 프레임마다)
  --no-show                 # Headless 모드
```

### 테스트

```bash
# ESP32 라이브 스트림 테스트
python tests/test_ocr_esp32.py --ip 192.168.0.100

# 이미지 OCR 후 서버 전송 테스트
python tests/test_ocr_image_send.py --image data/samples/Full1.jpeg --send-to 192.168.0.200

# 이미지 OCR만 (전송 없음)
python tests/test_ocr_image_send.py --image data/samples/Full1.jpeg
```

## 전송 JSON 형식

```json
{
  "timestamp": "2026-04-28 12:34:56",
  "status": "active",
  "results": {
    "pol1_temp_pv": "123.4",
    "iso_press": "2.5"
  }
}
```

## 학습

`colab_train.ipynb`를 Google Colab에서 열어 YOLOv8 학습을 진행합니다.  
데이터셋 전처리는 `prepare_dataset.py`를 사용합니다.
