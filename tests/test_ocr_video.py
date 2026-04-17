import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter, deque
import threading
import queue
import time

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ocr.compat  # noqa: F401  # PyTorch 2.6+ 호환성 패치
from ocr.easyocr_recognizer import EasyOcrRecognizer


class ValueStabilizer:
    """판독값의 급격한 변화를 막고 안정적인 값을 유지하기 위한 버퍼 클래스"""
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
        self.stable_value = ""

    def update(self, new_value):
        if new_value:  # 비어있지 않은 값만 버퍼에 추가
            self.buffer.append(new_value)

        if not self.buffer:
            return ""

        # 버퍼에서 가장 많이 등장한 값(최빈값)을 선택
        counts = Counter(self.buffer)
        most_common = counts.most_common(1)[0][0]
        self.stable_value = most_common
        return self.stable_value

def main():
    parser = argparse.ArgumentParser(description="OCR 비디오 정밀 테스트 (안정화 필터 적용)")
    parser.add_argument("--video", default=str(ROOT / 'data' / 'samples' / 'test_video.mp4'), help="입력 비디오 경로")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 감지 임계값")
    parser.add_argument("--skip", type=int, default=5, help="OCR 판독 간격 (영상은 매 프레임 재생)")
    args = parser.parse_args()

    # 1. 모델 및 인식기 초기화
    model = YOLO(str(ROOT / 'models' / 'best.pt'))
    recognizer = EasyOcrRecognizer()
    stabilizers = {}  # 변수별 안정화 필터 저장소
    last_results = {}  # 마지막 OCR 결과 보관용

    # OCR 백그라운드 처리를 위한 큐 및 워커 스레드 설정
    ocr_queue = queue.Queue(maxsize=10)  # 너무 많은 요청이 쌓이지 않게 제한

    def ocr_worker():
        while True:
            item = ocr_queue.get()
            if item is None: break  # 종료 신호

            crop, var_name = item
            try:
                # OCR 판독 (무거운 작업)
                raw_text = recognizer.read(crop, var_name=var_name)

                # 안정화 필터 적용 (필터 생성은 메인에서 보장)
                if var_name in stabilizers:
                    stable_text = stabilizers[var_name].update(raw_text)
                    last_results[var_name] = stable_text
            except Exception as e:
                print(f"[OCR Worker Error] {e}")
            finally:
                ocr_queue.task_done()

    worker_thread = threading.Thread(target=ocr_worker, daemon=True)
    worker_thread.start()

    # 비디오 경로 절대 경로로 변환
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {video_path}")
        return

    # 다중 백엔드 시도 (Mac용)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
        print("[System] 기본 드라이버 실패, FFmpeg 백엔드로 재시도합니다...")
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"[Error] 비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30  # 기본값
    frame_delay = int(1000 / fps)  # 프레임간 시간(ms)

    print(f"\n[System] 비디오 테스트 시작: {video_path}")
    print(f" - FPS: {fps}, Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(" - 'q': 종료, 'p': 일시정지")

    frame_count = 0
    consecutive_failures = 0
    paused = False

    while cap.isOpened():
        if not paused:
            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures < 5: continue
                print("비디오 재생이 완료되었거나 읽기에 실패했습니다.")
                break

            consecutive_failures = 0
            frame_count += 1
            display_frame = frame.copy()

            # 2. YOLO 감지 (매 프레임 수행하여 박스는 부드럽게)
            results = model.predict(frame, conf=args.conf, verbose=False)

            # OCR 판독 타이밍인지 확인
            is_ocr_step = (frame_count % args.skip == 0)

            # 3. 각 감지된 영역 처리
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                var_name = model.names[cls_id]

                # 안정화 필터 없으면 생성
                if var_name not in stabilizers:
                    stabilizers[var_name] = ValueStabilizer()

                if is_ocr_step:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        # OCR 요청을 큐에 전달 (비동기)
                        # 큐가 가득 찼다면 무시하여 지연 방지
                        try:
                            ocr_queue.put_nowait((crop.copy(), var_name))
                        except queue.Full:
                            pass

                # 시각화: 안정화된 값 표시
                current_val = last_results.get(var_name, "...")
                color = (0, 255, 0) if is_ocr_step else (200, 200, 200)  # 판독 시점에만 초록색 강조

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                label = f"{var_name}: {current_val}"
                # 텍스트 가독성을 위해 배경 상자 추가
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # 실시간 상태 표시
            status_text = f"Frame: {frame_count} | OCR Step: {'YES' if is_ocr_step else 'NO'}"
            cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 결과 화면 출력
            cv2.imshow("OCR Video Test (Stabilized)", display_frame)

        # 키 입력 처리 (FPS에 맞춘 딜레이 적용)
        key = cv2.waitKey(max(1, frame_delay - 5)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

    # 종료 처리
    ocr_queue.put(None)  # 워커 종료 신호
    worker_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("\n[System] 테스트 종료.")

if __name__ == "__main__":
    main()
