"""
YOLO 검출 + OCR 정량 평가 스크립트

사용법:
    # 검출 통계만 (빠름)
    python tests/eval_detection.py

    # OCR 결과까지 포함
    python tests/eval_detection.py --ocr

    # 정답 데이터를 추가해 정확도 측정
    python tests/eval_detection.py --ocr --ground-truth tests/ocr_ground_truth.json

정답 JSON 형식 (ocr_ground_truth.json):
    {
        "pol1_press":     "1.2",
        "pol2_press":     "0.8",
        "pol1_pump_speed": "42",
        ...
    }
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ocr.compat  # noqa: F401
from ocr.easyocr_recognizer import EasyOcrRecognizer


def compute_iou(box_a, box_b):
    """두 박스의 IoU를 계산합니다. box = (x1, y1, x2, y2)"""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_detection(model, img_path, conf):
    results = model.predict(str(img_path), conf=conf, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        crop = results[0].orig_img[y1:y2, x1:x2]
        detections.append({
            "class": cls_name,
            "conf": confidence,
            "box": (x1, y1, x2, y2),
            "crop": crop,
        })
    return detections


def print_detection_stats(detections):
    print("\n" + "=" * 55)
    print("  YOLO 검출 통계")
    print("=" * 55)

    by_class = defaultdict(list)
    for d in detections:
        by_class[d["class"]].append(d["conf"])

    total = len(detections)
    print(f"  총 검출 수: {total}개  |  클래스 수: {len(by_class)}개\n")
    print(f"  {'클래스':<25} {'검출수':>5}  {'평균신뢰도':>10}  {'최소':>6}  {'최대':>6}")
    print("  " + "-" * 53)
    for cls_name in sorted(by_class):
        confs = by_class[cls_name]
        print(f"  {cls_name:<25} {len(confs):>5}  {sum(confs)/len(confs):>10.3f}"
              f"  {min(confs):>6.3f}  {max(confs):>6.3f}")
    print("=" * 55)


def print_ocr_stats(detections, recognizer, ground_truth=None):
    print("\n" + "=" * 55)
    print("  OCR 판독 결과")
    print("=" * 55)

    results = {}
    for d in detections:
        cls_name = d["class"]
        ocr_val = recognizer.read(d["crop"], var_name=cls_name)
        results[cls_name] = ocr_val
        print(f"  {cls_name:<25}  →  '{ocr_val}'")

    if ground_truth:
        print("\n" + "=" * 55)
        print("  OCR 정확도 평가 (정답 데이터 기준)")
        print("=" * 55)
        correct = 0
        total = 0
        for cls_name, gt_val in ground_truth.items():
            pred_val = results.get(cls_name, "")
            match = "O" if pred_val == gt_val else "X"
            total += 1
            if pred_val == gt_val:
                correct += 1
            print(f"  [{match}] {cls_name:<25}  정답: '{gt_val}'  예측: '{pred_val}'")

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n  정확도: {correct}/{total}  ({accuracy:.1f}%)")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="YOLO 검출 + OCR 정량 평가")
    parser.add_argument("--image", default=str(ROOT / 'data' / 'samples' / 'Full1.jpeg'), help="평가할 이미지 경로")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO 신뢰도 임계값")
    parser.add_argument("--ocr", action="store_true", help="OCR 판독 결과 포함")
    parser.add_argument("--ground-truth", default=None, help="OCR 정답 JSON 파일 경로")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[Error] 이미지를 찾을 수 없습니다: {img_path}")
        sys.exit(1)

    print(f"[System] 모델 로딩 중...")
    model = YOLO(str(ROOT / 'models' / 'best.pt'))

    print(f"[System] 검출 실행 중: {img_path.name}")
    detections = run_detection(model, img_path, args.conf)

    print_detection_stats(detections)

    if args.ocr or args.ground_truth:
        recognizer = EasyOcrRecognizer()
        ground_truth = None
        if args.ground_truth:
            gt_path = Path(args.ground_truth)
            if not gt_path.exists():
                print(f"[Warning] 정답 파일을 찾을 수 없습니다: {gt_path}")
            else:
                with open(gt_path, encoding="utf-8") as f:
                    ground_truth = json.load(f)
        print_ocr_stats(detections, recognizer, ground_truth)


if __name__ == "__main__":
    main()
