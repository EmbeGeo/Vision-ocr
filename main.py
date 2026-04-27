import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import ocr.compat  # noqa: F401
from ocr.easyocr_recognizer import EasyOcrRecognizer
from ocr.stabilizer import ValueStabilizer
from ocr.video_client import VideoPullClient
from ocr.data_sender import DataSender


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("ocr_log.txt", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Vision OCR Raspberry Pi Main Service")
    parser.add_argument("--ip", type=str, required=True, help="ESP32-CAM의 IP 주소")
    parser.add_argument("--send-to", type=str, help="결과를 전송할 대상 IP")
    parser.add_argument("--target-port", type=int, default=8000, help="수신 서버 포트 (기본: 8000)")
    parser.add_argument("--target-path", type=str, default="/ocr", help="수신 서버 엔드포인트 (기본: /ocr)")
    parser.add_argument("--interval", type=int, default=10, help="전송 주기(초) (기본: 10)")
    parser.add_argument("--model", type=str, default="models/best.pt", help="YOLO 모델 경로")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 감지 임계값")
    parser.add_argument("--skip", type=int, default=5, help="OCR 판독 간격")
    parser.add_argument("--no-show", action="store_true", help="Headless 모드")
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 50)
    logging.info(" Vision OCR Service Starting...")
    logging.info(f" - ESP32 IP: {args.ip}")
    if args.send_to:
        logging.info(f" - Target IP: {args.send_to}")
    logging.info("=" * 50)

    logging.info("[System] Loading models...")
    try:
        model = YOLO(str(ROOT / args.model))
        recognizer = EasyOcrRecognizer()
    except Exception as e:
        logging.error(f"[Critical] Model load failed: {e}")
        return

    stabilizers = {}
    last_results = {}
    ocr_queue = queue.Queue(maxsize=5)

    def ocr_worker():
        while True:
            item = ocr_queue.get()
            if item is None:
                break
            crop, var_name = item
            try:
                raw_text = recognizer.read(crop, var_name=var_name)
                if var_name in stabilizers:
                    last_results[var_name] = stabilizers[var_name].update(raw_text)
            except Exception as e:
                logging.error(f"[OCR] worker error: {e}")
            finally:
                ocr_queue.task_done()

    threading.Thread(target=ocr_worker, daemon=True).start()

    receiver = VideoPullClient(args.ip).start()
    sender = None
    if args.send_to:
        sender = DataSender(
            target_ip=args.send_to,
            target_port=args.target_port,
            target_path=args.target_path,
            results_dict=last_results,
            interval=args.interval,
        )
        sender.start()

    frame_count = 0
    logging.info("[System] Service Loop Started. Press Ctrl+C to stop.")

    try:
        while True:
            frame = receiver.read()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1
            is_ocr_step = (frame_count % args.skip == 0)

            if is_ocr_step:
                results = model.predict(frame, conf=args.conf, verbose=False)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    var_name = model.names[int(box.cls[0])]

                    if var_name not in stabilizers:
                        stabilizers[var_name] = ValueStabilizer()

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        try:
                            ocr_queue.put_nowait((crop.copy(), var_name))
                        except queue.Full:
                            pass

            if not args.no_show:
                display_frame = frame.copy()
                for i, (var_name, val) in enumerate(last_results.items()):
                    cv2.putText(display_frame, f"{var_name}: {val}", (10, 30 + 20 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Vision OCR (Production)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        logging.info("[System] Shutdown signal received.")
    finally:
        receiver.stop()
        if sender:
            sender.stop()
        cv2.destroyAllWindows()
        logging.info("[System] Service terminated.")


if __name__ == "__main__":
    main()
