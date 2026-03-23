import time
import cv2


class ESP32Stream:

    def __init__(self, url: str, frame_skip: int = 2, reconnect_delay: float = 3.0):
        self.url = url
        self.frame_skip = frame_skip
        self.reconnect_delay = reconnect_delay
        self._cap = None

    def _connect(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def frames(self):
        count = 0
        while True:
            if self._cap is None or not self._cap.isOpened():
                print(f"[Capture] 접속 시도: {self.url}")
                self._cap = self._connect()
                if self._cap is None:
                    print(f"[Capture] 실패 → {self.reconnect_delay}초 후 재시도")
                    time.sleep(self.reconnect_delay)
                    continue
                print("[Capture] 접속 성공!")
            ret, frame = self._cap.read()
            if not ret:
                self._cap.release()
                self._cap = None
                continue
            count += 1
            if count % self.frame_skip == 0:
                yield frame

    def release(self):
        if self._cap:
            self._cap.release()
