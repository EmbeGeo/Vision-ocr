from ultralytics import YOLO
import cv2
import torch
import os
import sys

# 프로젝트 루트 경로 추가 (ocr 모듈 로드용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr.recognizer import DigitRecognizer

# PyTorch 2.6+ 보안 설정 해결
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

# ==================================================================
# 설정 (사용자 조절 가능)
# ==================================================================
CONF_THRESHOLD = 0.40
# ==================================================================

# 1. 모델 및 OCR 엔진 로드
model = YOLO('models/best.pt')
recognizer = DigitRecognizer()
names = model.names

def get_color(cls_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    return colors[cls_id % len(colors)]

# 2. 이미지 감지 실행
img_path = 'data/samples/Full1.png'
results = model.predict(img_path, conf=CONF_THRESHOLD)
img = results[0].orig_img.copy()

print(f"OCR 판독 시작...")

# 3. 박스 검출 및 OCR 적용
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    cls_name = names[cls_id]
    conf = float(box.conf[0])
    
    # 영역 잘라내기 (OCR용)
    crop = results[0].orig_img[y1:y2, x1:x2]
    
    # OCR 엔진으로 숫자 읽기
    ocr_res = recognizer.read(crop)
    
    # 결과 레이블 구성 (예: pol1_temp_pv: 15.3)
    display_text = f"{cls_name}: {ocr_res}"
    print(f"[{cls_name}] 감지됨 -> 판독값: {ocr_res} (신뢰도: {conf:.2f})")
    
    box_color = get_color(cls_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    
    # 텍스트 시각화
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # 레이블 위치 결정
    text_y = y2 + th + 10 if "sv" in cls_name.lower() else y1 - 10
    
    # 배경 및 텍스트 출력
    cv2.rectangle(img, (x1, text_y - th - 5), (x1 + tw + 5, text_y + baseline), box_color, -1)
    cv2.putText(img, display_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

# 4. 결과 출력
cv2.imshow("YOLO + Industrial 7-Segment OCR", img)
print("\n아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()
