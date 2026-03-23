import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from capture import ESP32Stream
from ocr import DisplayDetector, DigitReader


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def make_output(det, ocr) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bbox": list(det.bbox),
        "label": det.label,
        "value": ocr.value,
        "raw_text": ocr.raw_text,
        "confidence": round(ocr.confidence, 4),
    }


def run(source=None):
    cfg = load_config()

    detector = DisplayDetector(
        model_path=cfg["detect"]["model_path"],
        confidence=cfg["detect"]["confidence"],
        imgsz=cfg["detect"]["imgsz"],
    )
    reader = DigitReader(
        lang=cfg["ocr"]["lang"],
        use_angle_cls=cfg["ocr"]["use_angle_cls"],
        use_gpu=cfg["ocr"]["use_gpu"],
        show_log=cfg["ocr"]["show_log"],
    )

    out_cfg = cfg["output"]
    out_file = open(out_cfg["output_file"], "a") if out_cfg["save_to_file"] else None

    # source: None이면 ESP32 스트림, 파일 경로면 로컬 영상/이미지
    if source:
        cap = cv2.VideoCapture(source) if not source.endswith((".jpg",".png",".jpeg")) else None
        frames = _file_frames(source, cap)
    else:
        stream = ESP32Stream(**cfg["capture"])
        frames = stream.frames()

    print("실행 중... 종료: q")
    try:
        for frame in frames:
            detections = detector.detect(frame)
            for det in detections:
                results = reader.read(det.crop)
                for res in results:
                    output = make_output(det, res)
                    if out_cfg["print_result"]:
                        print(json.dumps(output, ensure_ascii=False))
                    if out_file:
                        out_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                        out_file.flush()

            vis = detector.draw(frame, detections)
            cv2.imshow("OCR Pipeline", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if out_file:
            out_file.close()
        cv2.destroyAllWindows()


def _file_frames(source: str, cap):
    # 이미지 파일이면 프레임 1장만 yield
    if source.endswith((".jpg", ".png", ".jpeg")):
        frame = cv2.imread(source)
        if frame is not None:
            yield frame
        return
    # 영상 파일이면 전체 프레임 yield
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


if __name__ == "__main__":
    # 인자 없으면 ESP32 스트림, 파일 경로 넘기면 로컬 테스트
    # python main.py
    # python main.py tests/samples/test.jpg
    # python main.py tests/samples/test.mp4
    source = sys.argv[1] if len(sys.argv) > 1 else None
    run(source)
