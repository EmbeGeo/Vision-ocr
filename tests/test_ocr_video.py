from ultralytics import YOLO
import cv2
import torch
import os
import sys

# ocr 모듈 로드 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr.recognizer import DigitRecognizer

# PyTorch 2.6+ 보안 설정 해결
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

# ==================================================================
# 설정 (사용자 조절 가능)
# ==================================================================
# 1. 신뢰도 임계값: 너무 민감하면 값을 높이세요 (0.45), 안 잡히면 낮추세요 (0.30)
CONF_THRESHOLD = 0.40
# 2. 비디오 경로: video.mp4 또는 video_compat.mp4
VIDEO_PATH = "data/samples/video_compat.mp4"
# ==================================================================

# 1. 모델 및 OCR 엔진 로드
model = YOLO('models/best.pt')
recognizer = DigitRecognizer()
names = model.names

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"오류: 비디오 파일({VIDEO_PATH})을 열 수 없습니다.")
    sys.exit()

print(f"비디오 실시간 OCR 분석 시작...")
print("Q 키를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 2. YOLO 프레임 감지
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    
    # 3. 결과 시각화 및 OCR
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        
        # OCR 수행용 크롭 영역
        crop = frame[y1:y2, x1:x2]
        ocr_res = recognizer.read(crop, var_name=cls_name)
        
        box_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)][cls_id % 6]
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # 화면 표시 텍스트 (예: pol1_temp_pv: 15.3)
        display_text = f"{cls_name}: {ocr_res}"
        
        # 텍스트 시각화 설정
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_y = y2 + th + 10 if "sv" in cls_name.lower() else y1 - 10
        cv2.rectangle(frame, (x1, text_y - th - 5), (x1 + tw + 5, text_y + baseline), box_color, -1)
        cv2.putText(frame, display_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # 4. 화면 출력 및 종료 제어
    cv2.imshow("Video OCR Real-time Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
