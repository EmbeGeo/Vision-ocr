# 📊 Dashboard Variable Detection (YOLOv8)

이 프로젝트는 산업용 계기판(대시보드)에서 각종 변수(PV, SV, 압축기 상태 등)를 자동으로 감지하기 위한 YOLOv8 기반의 딥러닝 솔루션입니다.

## 📁 주요 폴더 및 파일 구조

-   `data/`
    -   `dataset/`: YOLOv8 학습을 위한 정규화된 데이터셋 (이미지 및 라벨)
    -   `samples/`: 데이터 증강의 기반이 되는 원본 이미지(`Full1.png`)와 VOC XML(`Full1.xml`)
-   `models/`: 학습 완료된 가중치 파일 (`best.pt`) 보관
-   `tests/`: 모델의 성능을 즉시 확인할 수 있는 테스트 스크립트
    -   `test_image.py`: 단일 이미지 내 변수 감지 및 시각화
    -   `test_video.py`: 비디오 파일 실시간 감지 및 시각화
-   `prepare_dataset.py`: VOC XML 기반의 자동 데이터 증강(Augmentation) 및 데이터셋 패키징 도구
-   `colab_train.ipynb`: 구글 코랩(Google Colab)에서 원클릭으로 학습을 진행할 수 있는 노트북
-   `requirements.txt`: 프로젝트 실행에 필요한 파이썬 패키지 목록
-   `dataset.zip`: 코랩 학습을 위해 준비된 압축 데이터셋

## 🛠️ 환경 구축 (Setup)

가상환경을 생성하고 필요한 패키지를 설치합니다.

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 필수 패키지 설치
pip install -r requirements.txt
```

## 🚀 사용법 (Usage)

### 1. 감지 테스트
학습된 `models/best.pt` 가중치를 사용하여 감지 결과를 확인합니다.

```bash
# 이미지 테스트 (윈도우 창으로 결과 표시)
python tests/test_image.py

# 비디오 테스트 (실시간 재생 모드)
python tests/test_video.py
```

### 2. 데이터 증강 및 학습 준비
적은 수의 샘플로 대량의 학습 데이터를 생성하고 코랩용 패키지를 만듭니다.

```bash
# 실행 시 data/dataset을 200장 규모로 채우고 dataset.zip을 생성합니다.
python prepare_dataset.py
```

### 3. 모델 학습 (Google Colab)
1. `colab_train.ipynb`를 구글 코랩에서 엽니다.
2. `dataset.zip` 파일을 업로드한 후 노트북의 안내에 따라 학습을 진행합니다.

## 💡 주요 특징
-   **가독성 특화 시각화**: `sv` 변수의 레이블은 값을 가리지 않도록 하단에 표기됩니다.
-   **오탐지 제어**: `test_image.py` 등의 상단에서 `CONF_THRESHOLD`를 조절하여 감도를 변경할 수 있습니다.
-   **변수별 색상**: 각 변수 클래스마다 고유한 색상의 박스가 할당되어 구분이 쉽습니다.
