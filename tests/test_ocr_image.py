import sys
import os
from pathlib import Path

from ultralytics import YOLO
import cv2
import numpy as np

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ocr.compat  # noqa: F401  # PyTorch 2.6+ 호환성 패치
from ocr.easyocr_recognizer import EasyOcrRecognizer

# ==================================================================
# 설정 (사용자 조절 가능)
# ==================================================================
CONF_THRESHOLD = 0.40
# ==================================================================

# 1. 모델 및 OCR 엔진 로드
model = YOLO(str(ROOT / 'models' / 'best.pt'))
recognizer = EasyOcrRecognizer()
names = model.names

def get_color(cls_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    return colors[cls_id % len(colors)]

# 2. 이미지 감지 실행
img_path = str(ROOT / 'data' / 'samples' / 'Full1.jpeg')
results = model.predict(img_path, conf=CONF_THRESHOLD)
img = results[0].orig_img.copy()
h, w, c = img.shape

# 결과 리스트업을 위한 사이드 패널 생성
side_panel_w = 400
side_panel = np.ones((h, side_panel_w, 3), dtype=np.uint8) * 40  # Dark grey background
cv2.putText(side_panel, "OCR Results Panel", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
y_offset = 80

print(f"OCR 판독 시작...")

# 3. 박스 검출 및 OCR 적용
for idx, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    cls_name = names[cls_id]
    conf = float(box.conf[0])

    # 영역 잘라내기 (OCR용)
    crop = results[0].orig_img[y1:y2, x1:x2]

    # OCR 엔진으로 숫자 읽기 (디버깅을 위해 클래스 이름 전달)
    ocr_res = recognizer.read(crop, var_name=cls_name, box_idx=idx+1)

    box_color = get_color(cls_id)

    # 이미지에는 심플하게 바운딩 박스와 식별용 번호표만 표기 (간섭 방지)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    label_num = f"[{idx+1}]"

    # 라벨을 박스 근처에 텍스트만 표시 (배경 채움 제거하여 다른 세그먼트 가림 완벽 방지)
    text_y = y1 - 5 if y1 >= 22 else y2 + 15
    # 테두리 효과 (가독성 확보)
    cv2.putText(img, label_num, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, label_num, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # 사이드 패널에 변수명 및 판독 결과 깔끔하게 출력
    display_text = f"{label_num} {cls_name}: {ocr_res}"
    print(f"{label_num} [{cls_name}] 감지됨 -> 판독값: {ocr_res} (신뢰도: {conf:.2f})")

    font_scale = 0.5
    thickness = 1
    # 컬러 박스 (범례 역할)
    cv2.rectangle(side_panel, (15, y_offset - 12), (22, y_offset - 5), box_color, -1)
    # 텍스트 출력
    cv2.putText(side_panel, display_text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    y_offset += 30

# 붙이기
final_img = np.hstack((img, side_panel))

# 4. 결과 출력
cv2.imshow("Dashboard Monitor & Output", final_img)
print("\n아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()
