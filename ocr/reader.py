import re
from dataclasses import dataclass

import cv2
import numpy as np
from paddleocr import PaddleOCR


@dataclass
class OCRResult:
    raw_text: str
    value: float | None  # float 변환 실패 시 None
    confidence: float


class DigitReader:

    def __init__(self, lang="en", use_angle_cls=False, use_gpu=False, show_log=False):
        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls, lang=lang,
            use_gpu=use_gpu, show_log=show_log,
        )

    @staticmethod
    def _parse(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            return None

    def read(self, crop: np.ndarray) -> list[OCRResult]:
        # 원본 이미지 통째로 PaddleOCR 딥러닝 모델에 위임 (강제 이진화 시 거대 해상도에서 선이 깨짐)
        result = self._ocr.ocr(crop, cls=False)
        if not result or not result[0]:
            return []
            
        found_texts = []
        for line in result[0]:
            raw, conf = line[1][0], float(line[1][1])
            val = self._parse(raw)
            if val is not None:
                found_texts.append(OCRResult(raw_text=raw, value=val, confidence=conf))
        return found_texts
