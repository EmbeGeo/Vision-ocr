import argparse
import os
import sys

import cv2
import torch
from ultralytics import YOLO

# 프로젝트 루트 경로 추가 (OCR 옵션 사용 시)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PyTorch 2.6+ 보안 설정(weights_only)으로 인한 모델 로딩 실패 해결
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "weights_only": False})


def get_color(cls_id):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 165, 0),
        (255, 20, 147),
        (0, 250, 154),
        (30, 144, 255),
        (127, 255, 0),
        (139, 69, 19),
    ]
    return colors[cls_id % len(colors)]


def main():
    parser = argparse.ArgumentParser(description="YOLO 비디오 감지 테스트")
    parser.add_argument("--video", default="data/samples/test_video.mp4", help="입력 비디오 경로")
    parser.add_argument("--conf", type=float, default=0.40, help="감지 confidence threshold")
    parser.add_argument("--ocr", action="store_true", help="박스 내부 값 OCR까지 함께 수행")
    args = parser.parse_args()

    model = YOLO("models/best.pt")
    names = model.names

    recognizer = None
    if args.ocr:
        from ocr.easyocr_recognizer import EasyOcrRecognizer
        recognizer = EasyOcrRecognizer()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: {video_path} 비디오 파일을 찾을 수 없습니다.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} 비디오 파일을 열 수 없습니다.")
        return

    print("'q' 키를 누르면 비디오 창을 종료합니다.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("비디오 재생이 완료되었습니다.")
            break

        results = model.predict(frame, conf=args.conf, verbose=False)

        fh, fw = frame.shape[:2]
        font_scale = 0.4
        text_thickness = 1

        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])

            if recognizer is None:
                label = f"{cls_name} ({conf:.2f})"
            else:
                crop = frame[y1:y2, x1:x2]
                ocr_res = recognizer.read(crop, var_name=cls_name)
                label = f"{cls_name}: {ocr_res}"
            dets.append((x1, y1, x2, y2, cls_id, cls_name, label))

        for i, (x1, y1, x2, y2, cls_id, cls_name, label) in enumerate(dets):
            box_color = get_color(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # 라벨을 바운딩 박스 밖에 배치 (세그먼트 가림 방지)
            if y1 >= 22:
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + int(len(label)*8.5), y1), box_color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)
            else:
                cv2.rectangle(frame, (x1, y2), (x1 + int(len(label)*8.5), y2 + 20), box_color, -1)
                cv2.putText(frame, label, (x1 + 3, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)

        window_name = "YOLO Video Test + OCR" if args.ocr else "YOLO Video Test"
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
