import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import threading
import queue
import time
from collections import Counter, deque
from ultralytics import YOLO

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ocr.compat  # noqa: F401  # PyTorch 2.6+ 호환성 패치
from ocr.easyocr_recognizer import EasyOcrRecognizer


class VideoPullClient:
    """ESP32-CAM의 HTTP MJPEG 스트림에 접속하여 최신 프레임을 유지하는 클라이언트"""
    def __init__(self, ip, port=80, stream_path="/stream"):
        if ip.startswith("http://") or ip.startswith("https://"):
            self.url = ip
        else:
            self.url = f"http://{ip}:{port}{stream_path}"
        self.frame = None
        self.stopped = False
        self.status = f"Connecting to {self.url}..."
        self.cap = None

    def start(self):
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        return self

    def _run_client(self):
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.status = f"Connecting to {self.url}..."
                    self.cap = cv2.VideoCapture(self.url)
                    # 타임아웃 설정을 위해 FFmpeg 전용 옵션을 고려할 수 있으나 기본으로 시작
                
                success, frame = self.cap.read()
                if success:
                    self.frame = frame
                    self.status = "Streaming Active"
                else:
                    self.status = "Stream lost. Reconnecting..."
                    self.cap.release()
                    time.sleep(1.0)
            except Exception as e:
                self.status = f"Error: {e}"
                time.sleep(2.0)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

class ValueStabilizer:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
        self.stable_value = ""

    def update(self, new_value):
        if new_value:
            self.buffer.append(new_value)
        if not self.buffer: return ""
        counts = Counter(self.buffer)
        self.stable_value = counts.most_common(1)[0][0]
        return self.stable_value

def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM OCR 수신 클라이언트 (Pull)")
    parser.add_argument("--ip", type=str, required=True, help="ESP32-CAM의 IP 주소")
    parser.add_argument("--port", type=int, default=80, help="ESP32 웹 서버 포트 (기본: 80)")
    parser.add_argument("--path", type=str, default="/stream", help="스트림 경로 (기본: /stream)")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 감지 임계값")
    parser.add_argument("--skip", type=int, default=5, help="OCR 판독 간격")
    args = parser.parse_args()

    # 1. 모델 로딩
    print("[System] 모델 로딩 중...")
    try:
        model = YOLO(str(ROOT / 'models' / 'best.pt'))
        recognizer = EasyOcrRecognizer()
    except Exception as e:
        print(f"[Error] 모델 로드 실패: {e}")
        return

    stabilizers = {}
    last_results = {}
    ocr_queue = queue.Queue(maxsize=5)
    
    def ocr_worker():
        while True:
            item = ocr_queue.get()
            if item is None: break
            crop, var_name = item
            try:
                raw_text = recognizer.read(crop, var_name=var_name)
                if var_name in stabilizers:
                    last_results[var_name] = stabilizers[var_name].update(raw_text)
            finally:
                ocr_queue.task_done()

    threading.Thread(target=ocr_worker, daemon=True).start()

    # 2. 클라이언트 시작
    receiver = VideoPullClient(args.ip, args.port, args.path).start()
    
    print("=" * 50)
    print(f" [ESP32 접속 정보]")
    print(f" 최종 URL: {receiver.url}")
    print("=" * 50)
    
    frame_count = 0
    print("\n[System] 스트림 수신 대기 시작 (종료: 'q')")
    
    try:
        while True:
            frame = receiver.read()
            if frame is None:
                # 대기 화면
                bg = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(bg, receiver.status, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(bg, f"Connecting to http://{args.ip}:{args.port}{args.path}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("ESP32 OCR Server", bg)
                if cv2.waitKey(100) & 0xFF == ord('q'): break
                continue

            frame_count += 1
            display_frame = frame.copy()
            
            # YOLO & OCR
            results = model.predict(frame, conf=args.conf, verbose=False)
            is_ocr_step = (frame_count % args.skip == 0)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                var_name = model.names[int(box.cls[0])]
                
                if var_name not in stabilizers: stabilizers[var_name] = ValueStabilizer()
                
                if is_ocr_step:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        try: ocr_queue.put_nowait((crop.copy(), var_name))
                        except queue.Full: pass
                
                val = last_results.get(var_name, "...")
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{var_name}: {val}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(display_frame, "Streaming Active", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("ESP32 OCR Server", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    finally:
        receiver.stop()
        cv2.destroyAllWindows()
        print("[System] 종료.")

if __name__ == "__main__":
    main()
