import cv2
import numpy as np
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class DigitRecognizer:
    """산업용 7세그먼트 전용 OCR 엔진 (Template Matching + Multi-threshold)

    .. deprecated::
        EasyOcrRecognizer로 대체되었습니다. ocr/legacy/ 폴더에 보관됩니다.
    """

    def __init__(self, debug=True):
        self.debug = debug
        self.debug_dir = str(_PROJECT_ROOT / "data" / "debug_ocr")
        if self.debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            
        # 7세그먼트 표준 맵핑 (A, B, C, D, E, F, G)
        self.segments_map = {
            (1, 1, 1, 1, 1, 1, 0): "0",
            (0, 1, 1, 0, 0, 0, 0): "1",
            (1, 1, 0, 1, 1, 0, 1): "2",
            (1, 1, 1, 1, 0, 0, 1): "3",
            (0, 1, 1, 0, 0, 1, 1): "4",
            (1, 0, 1, 1, 0, 1, 1): "5",
            (1, 0, 1, 1, 1, 1, 1): "6",
            (1, 1, 1, 0, 0, 0, 0): "7",
            (1, 1, 1, 1, 1, 1, 1): "8",
            (1, 1, 1, 1, 0, 1, 1): "9",
            (0, 0, 0, 0, 0, 0, 1): "-",
            (0, 0, 0, 0, 0, 0, 0): ""
        }

    def get_binary_image(self, crop, thresh_val):
        """다양한 임계값으로 이진화 수행"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 산업용 디바이스: 숫자가 밝고 배경이 어두운 경우 (THRESH_BINARY)
        # 또는 배경이 밝고 숫자가 어두운 경우 (THRESH_BINARY_INV) 선택 가능
        # 여기서는 숫자가 상대적으로 밝다고 가정하여 THRESH_BINARY 사용
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        return thresh

    def split_digits(self, thresh):
        """수직 투영을 이용한 숫자 분리"""
        h, w = thresh.shape
        projection = np.sum(thresh, axis=0)
        
        digit_zones = []
        start = None
        
        # 최소 숫자 너비 설정 (노이즈 방지)
        min_width = max(2, w // 20)
        
        for x in range(w):
            if projection[x] > 0 and start is None:
                start = x
            elif projection[x] == 0 and start is not None:
                if (x - start) >= min_width:
                    digit_zones.append((start, x))
                start = None
        if start is not None and (w - start) >= min_width:
            digit_zones.append((start, w))
            
        return digit_zones

    def recognize_zonal(self, digit_img):
        """7개 구역 분석 (Zonal Analysis) - 개선된 버전"""
        h, w = digit_img.shape
        if w < 3 or h < 5: return ""
        
        # 구역 설정 (상대 좌표 비율)
        # 상(A), 우상(B), 우하(C), 하(D), 좌하(E), 좌상(F), 중(G)
        zones = [
            (h//10, w//2),        # A
            (h//4, w*8//10),     # B
            (h*3//4, w*8//10),   # C
            (h*9//10, w//2),     # D
            (h*3//4, w*2//10),   # E
            (h//4, w*2//10),     # F
            (h//2, w//2)         # G
        ]
        
        active = []
        for r_y, r_x in zones:
            # 해당 좌표 주변 ROI의 평균값 확인
            rw, rh = max(1, w//8), max(1, h//12)
            y1, y2 = max(0, r_y-rh), min(h, r_y+rh)
            x1, x2 = max(0, r_x-rw), min(w, r_x+rw)
            roi = digit_img[y1:y2, x1:x2]
            
            if roi.size > 0 and np.mean(roi) > 50: # 활성화 임계값
                active.append(1)
            else:
                active.append(0)
                
        return self.segments_map.get(tuple(active), "?")

    def read(self, crop, var_name="unknown", box_idx=None):
        """앙상블 판독 (여러 임계값으로 시도하여 다수결 선택)"""
        if crop is None or crop.size == 0: return ""
        
        results = []
        # 다양한 임계값으로 시도 (100 ~ 200 범위)
        thresholds = [100, 130, 160, 190]
        
        for idx, t_val in enumerate(thresholds):
            thresh = self.get_binary_image(crop, t_val)
            zones = self.split_digits(thresh)
            
            current_read = ""
            for i, (sx, ex) in enumerate(zones):
                digit_img = thresh[:, sx:ex]
                
                # 소수점 감지 로직 (폭이 좁고 하단에 위치)
                if (ex - sx) < (thresh.shape[1] // 12) and np.sum(digit_img[int(thresh.shape[0]*0.7):, :]) > 0:
                    current_read += "."
                else:
                    digit_val = self.recognize_zonal(digit_img)
                    current_read += digit_val
            
            if current_read:
                results.append(current_read)
            
            # 디버그 이미지 저장
            if self.debug:
                cv2.imwrite(f"{self.debug_dir}/{var_name}_t{t_val}.png", thresh)
        
        if not results: return ""
        
        # 가장 많이 등장한 결과 반환 (투표)
        from collections import Counter
        most_common = Counter(results).most_common(1)
        return most_common[0][0] if most_common else ""
