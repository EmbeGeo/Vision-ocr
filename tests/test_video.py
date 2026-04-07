from ultralytics import YOLO
import cv2
import torch

# PyTorch 2.6+ 보안 설정(weights_only)으로 인한 모델 로딩 실패 해결을 위한 몽키패치
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

# ==================================================================
# 설정 (사용자 조절 가능)
# ==================================================================
# 1. 신뢰도 임계값 (0.0 ~ 1.0)
# - 값이 높을수록(예: 0.50) 확실한 것만 감지하여 오탐지가 줄어듭니다.
# - 감지가 너무 안 된다면 값을 낮춰보세요(예: 0.25).
CONF_THRESHOLD = 0.40
# ==================================================================

# 1. 모델 로드 (models/best.pt)
model = YOLO('models/best.pt')
names = model.names

def get_color(cls_id):
    """클래스 고유 색상을 반환 (BGR 형식)"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 165, 0), (255, 20, 147), (0, 250, 154), (30, 144, 255), (127, 255, 0), (139, 69, 19)
    ]
    return colors[cls_id % len(colors)]

# 2. 비디오 파일 열기 (data/samples/video_compat.mp4)
video_path = 'data/samples/video_compat.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: {video_path} 비디오 파일을 열 수 없습니다.")
    exit()

# 3. 비디오 프레임 루프
print("'q' 키를 누르면 비디오 창을 종료합니다.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("비디오 재생이 완료되었습니다.")
        break

    # 4. YOLOv8 감지 실행 (임계값 적용)
    results = model.predict(frame, conf=CONF_THRESHOLD)
    
    # 5. 직접 시각화 (가독성 개선 및 레이블 위치 조정)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf = float(box.conf[0])
        
        label = f"{cls_name} ({conf:.2f})"
        
        # 변수별 고유 색상 가져오기
        box_color = get_color(cls_id)
        
        # 1. 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # 2. 폰트 및 텍스트 설정
        font_scale = 0.4
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        
        # 3. 레이블 위치 결정: 'sv'가 포함되어 있으면 하단, 아니면 상단
        if "sv" in cls_name.lower():
            text_y = y2 + th + 5
        else:
            text_y = y1 - 5
            
        # 4. 텍스트 배경 (박스 색상과 동일하게 설정하여 구분)
        cv2.rectangle(frame, (x1, text_y - th - 5), (x1 + tw, text_y + baseline), box_color, -1)
        
        # 5. 텍스트 쓰기 (검은색)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)

    # 6. 결과 프레임 화면 표시
    cv2.imshow("Custom YOLOv8 Detection - Multi-Color", frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. 리소스 해제
cap.release()
cv2.destroyAllWindows()
