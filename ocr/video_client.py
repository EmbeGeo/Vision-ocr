import time
import logging
import threading

import cv2


class VideoPullClient:
    """ESP32-CAM의 HTTP MJPEG 스트림에 접속하여 최신 프레임을 유지하는 클라이언트"""
    def __init__(self, ip, port=80, stream_path="/stream"):
        if ip.startswith("http://") or ip.startswith("https://"):
            self.url = ip
        else:
            self.url = f"http://{ip}:{port}{stream_path}"
        self.frame = None
        self.stopped = False
        self.status = "Initializing..."
        self.cap = None

    def start(self):
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        return self

    def _run_client(self):
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logging.info(f"[Video] Connecting to {self.url}...")
                    self.cap = cv2.VideoCapture(self.url)

                success, frame = self.cap.read()
                if success:
                    self.frame = frame
                    self.status = "Streaming Active"
                else:
                    logging.warning("[Video] Stream lost. Reconnecting...")
                    self.cap.release()
                    time.sleep(1.0)
            except Exception as e:
                logging.error(f"[Video] Connection Error: {e}")
                time.sleep(2.0)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()
