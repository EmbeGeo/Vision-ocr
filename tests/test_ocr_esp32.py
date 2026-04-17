import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import socket
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

def get_local_ip():
    """PC의 로컬 IP 주소를 가져옵니다."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class VideoReceiverServer:
    """ESP32-CAM에서 보낸 영상을 받아 최신 프레임을 유지하는 소켓 서버"""
    def __init__(self, port=5000):
        self.port = port
        self.frame = None
        self.stopped = False
        self.status = "Waiting for connection..."
        self.server_socket = None

    def start(self):
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        return self

    def _run_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0) # 주기적으로 stopped 체크를 위해

            print(f"[Receiver] 서버 대기 중... 포트: {self.port}")
            
            while not self.stopped:
                try:
                    conn, addr = self.server_socket.accept()
                    print(f"[Receiver] 연결됨: {addr}")
                    self.status = f"Connected to {addr[0]}"
                    self._handle_client(conn)
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stopped:
                        print(f"[Receiver] Accept Error: {e}")
        except Exception as e:
            self.status = f"Server Error: {e}"
            print(f"[Receiver] {self.status}")

    def _handle_client(self, conn):
        bytes_data = b''
        try:
            while not self.stopped:
                chunk = conn.recv(4096)
                if not chunk:
                    print("[Receiver] 연결 종료")
                    break
                
                bytes_data += chunk
                # MJPEG/JPEG 스타일의 경계 탐색 (FFD8 ~ FFD9)
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    new_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if new_frame is not None:
                        self.frame = new_frame
        except Exception as e:
            print(f"[Receiver] Client Handler Error: {e}")
        finally:
            conn.close()
            self.status = "Disconnected. Waiting..."

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.server_socket:
            self.server_socket.close()

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
    parser = argparse.ArgumentParser(description="ESP32-CAM OCR 수신 서버")
    parser.add_argument("--port", type=int, default=5000, help="수신 포트 (기본: 5000)")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 감지 임계값")
    parser.add_argument("--skip", type=int, default=5, help="OCR 판독 간격")
    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 50)
    print(f" [ESP32 설정용 정보]")
    print(f" PC IP 주소: {local_ip}")
    print(f" 포트 번호: {args.port}")
    print(f" ESP32 코드에 위 주소를 입력하여 데이터를 전송하세요.")
    print("=" * 50)

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
    
    # 2. 서버 시작
    receiver = VideoReceiverServer(args.port).start()
    
    frame_count = 0
    print("\n[System] 수신 대기 시작 (종료: 'q')")
    
    try:
        while True:
            frame = receiver.read()
            if frame is None:
                # 대기 화면
                bg = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(bg, receiver.status, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(bg, f"IP: {local_ip} | Port: {args.port}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
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
