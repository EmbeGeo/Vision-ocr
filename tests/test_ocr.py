# OCR만 테스트 (이미지)
# python tests/test_ocr.py --mode ocr --source tests/samples/display.jpg

# YOLO 탐지 테스트 (이미지 또는 영상)
# python tests/test_ocr.py --mode detect --source tests/samples/recording.mp4

# 전체 파이프라인 테스트
# python tests/test_ocr.py --mode all --source tests/samples/recording.mp4


import argparse
import sys
import cv2
import yaml

sys.path.insert(0, ".")
from ocr import DisplayDetector, DigitReader


def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def iter_frames(source: str):
    if source.endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"이미지 로드 실패: {source}")
        yield frame
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"영상 로드 실패: {source}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()


def test_ocr(source: str):
    # OCR 단독 테스트 - YOLO 없이 이미지 전체를 OCR에 넣어봄
    cfg = load_cfg()["ocr"]
    reader = DigitReader(**cfg)
    print(f"\n[OCR 테스트] {source}")
    for i, frame in enumerate(iter_frames(source)):
        result = reader.read(frame)
        print(f"  프레임 {i:03d}: {result}")
        cv2.imshow("OCR Test", frame)
        if cv2.waitKey(300) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def test_detect(source: str):
    # YOLO 탐지 테스트 - bbox를 화면에 그려서 확인
    cfg = load_cfg()
    detector = DisplayDetector(
        model_path=cfg["detect"]["model_path"],
        confidence=cfg["detect"]["confidence"],
        imgsz=cfg["detect"]["imgsz"],
    )
    print(f"\n[YOLO 탐지 테스트] {source}  |  종료: q")
    for frame in iter_frames(source):
        dets = detector.detect(frame)
        vis = detector.draw(frame, dets)
        if dets:
            print(f"  탐지: {[f'{d.label}({d.confidence:.2f})' for d in dets]}")
        cv2.imshow("Detect Test", vis)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def test_all(source: str):
    # 전체 파이프라인 테스트 - YOLO → OCR 결과 출력
    cfg = load_cfg()
    detector = DisplayDetector(
        model_path=cfg["detect"]["model_path"],
        confidence=cfg["detect"]["confidence"],
        imgsz=cfg["detect"]["imgsz"],
    )
    reader = DigitReader(**cfg["ocr"])
    print(f"\n[전체 파이프라인 테스트] {source}  |  종료: q")
    for frame in iter_frames(source):
        dets = detector.detect(frame)
        for det in dets:
            results = reader.read(det.crop)
            for res in results:
                print(f"  bbox={det.bbox}  value={res.value}  "
                      f"raw='{res.raw_text}'  conf={res.confidence:.2f}")
        cv2.namedWindow("Pipeline Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pipeline Test", 1024, 768)
        cv2.imshow("Pipeline Test", detector.draw(frame, dets))
        wait_time = 0 if source.endswith((".jpg", ".jpeg", ".png")) else 1
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ocr", "detect", "all"], default="all")
    parser.add_argument("--source", required=True, help="이미지 또는 영상 파일 경로")
    args = parser.parse_args()

    {"ocr": test_ocr, "detect": test_detect, "all": test_all}[args.mode](args.source)
