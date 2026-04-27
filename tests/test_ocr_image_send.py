import sys
import json
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ocr.compat  # noqa: F401
from ocr.easyocr_recognizer import EasyOcrRecognizer

CONF_THRESHOLD = 0.40


def send_json(url: str, payload: dict) -> bool:
    json_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=json_data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[Sender] {resp.status} - 전송 성공")
            return True
    except Exception as e:
        print(f"[Sender] 전송 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="이미지 OCR 결과를 JSON으로 전송하는 테스트")
    parser.add_argument("--image", type=str, default=str(ROOT / "data" / "samples" / "Full1.jpeg"),
                        help="테스트할 이미지 경로")
    parser.add_argument("--send-to", type=str, default=None,
                        help="JSON을 전송할 대상 IP (예: 192.168.1.100)")
    parser.add_argument("--port", type=int, default=8000, help="수신 서버 포트 (기본: 8000)")
    parser.add_argument("--path", type=str, default="/ocr", help="수신 서버 엔드포인트 (기본: /ocr)")
    args = parser.parse_args()

    print("[System] 모델 로딩 중...")
    model = YOLO(str(ROOT / "models" / "best.pt"))
    recognizer = EasyOcrRecognizer()
    names = model.names

    print(f"[System] 이미지 분석: {args.image}")
    results = model.predict(args.image, conf=CONF_THRESHOLD, verbose=False)

    ocr_results = {}

    print("OCR 판독 시작...")
    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = names[int(box.cls[0])]
        conf = float(box.conf[0])

        crop = results[0].orig_img[y1:y2, x1:x2]
        ocr_val = recognizer.read(crop, var_name=cls_name, box_idx=idx + 1)

        ocr_results[cls_name] = ocr_val
        print(f"  [{cls_name}] -> {ocr_val} (conf: {conf:.2f})")

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active",
        "results": ocr_results,
    }

    print("\n[JSON 페이로드]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.send_to:
        url = f"http://{args.send_to}:{args.port}{args.path}"
        print(f"\n[Sender] 전송 대상: {url}")
        send_json(url, payload)
    else:
        print("\n[Sender] --send-to 옵션 없음 — 전송 생략 (로컬 출력만)")


if __name__ == "__main__":
    main()
