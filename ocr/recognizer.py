import cv2
import numpy as np

class DigitRecognizer:
    """산업용 7세그먼트 전용 OCR 엔진"""
    
    def __init__(self):
        # 7세그먼트 맵핑 (A, B, C, D, E, F, G)
        # 상, 우상, 우하, 하, 좌하, 좌상, 중
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
            (0, 0, 0, 0, 0, 0, 1): "-", # 마이너스 기호
            (0, 0, 0, 0, 0, 0, 0): ""   # 빈 칸
        }

    def preprocess(self, crop):
        """기본 이진화 처리 (엔진 구동을 위한 최소 전처리)"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 밝은 부분(세그먼트)을 흰색(255)으로 만들기 위해 OTSU + INV 또는 일반 이진화
        # 배경보다 숫자가 밝은 경우를 가정 (산업 계기판 일반적)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 만약 배경이 더 밝다면 반전 필요 (체크 로직 필요할 수 있음)
        # 여기서는 숫자가 밝다고 가정하고 진행
        return thresh

    def split_digits(self, thresh):
        """수직 투영을 통해 개별 숫자를 분리"""
        height, width = thresh.shape
        projection = np.sum(thresh, axis=0)
        
        # 임계값 이상의 픽셀이 있는 구간을 숫자로 판단
        digit_zones = []
        in_digit = False
        start_x = 0
        
        for x in range(width):
            if projection[x] > 0 and not in_digit:
                start_x = x
                in_digit = True
            elif projection[x] == 0 and in_digit:
                if x - start_x > 2: # 너무 좁은 구간은 노이즈로 간주
                    digit_zones.append((start_x, x))
                in_digit = False
        
        if in_digit:
            digit_zones.append((start_x, width))
            
        return digit_zones

    def recognize_digit(self, digit_img):
        """하나의 숫자 이미지에서 7세그먼트 영역을 분석하여 판독"""
        h, w = digit_img.shape
        if w < 3 or h < 5: return "" # 너무 작은 영역 제외
        
        # 7개 영역 좌표 정의 (상대적 비율)
        # h=높이, w=너비
        mw, mh = w // 2, h // 2 # 중간 지점
        offset = 2 # 테두리 오프셋
        
        # 각 세그먼트의 중심점 샘플링
        segments = [
            (mw, offset),           # A (상)
            (w - offset, h // 4),   # B (우상)
            (w - offset, 3 * h // 4),# C (우하)
            (mw, h - offset),       # D (하)
            (offset, 3 * h // 4),   # E (좌하)
            (offset, h // 4),       # F (좌상)
            (mw, mh)                # G (중)
        ]
        
        active = []
        for sx, sy in segments:
            # 해당 좌표 주변 3x3 영역의 평균 밝기를 체크
            roi = digit_img[max(0, sy-1):min(h, sy+2), max(0, sx-1):min(w, sx+2)]
            if np.mean(roi) > 127: # 켜져 있다고 판단
                active.append(1)
            else:
                active.append(0)
        
        return self.segments_map.get(tuple(active), "?")

    def read(self, crop):
        """전체 변수 박스를 입력받아 최종 문자열 반환"""
        if crop is None or crop.size == 0: return ""
        
        thresh = self.preprocess(crop)
        digit_zones = self.split_digits(thresh)
        
        result = ""
        for start_x, end_x in digit_zones:
            digit_img = thresh[:, start_x:end_x]
            # 소수점 감지 로직 (높이가 낮고 너비가 좁은 경우)
            if (end_x - start_x) < (thresh.shape[1] // 10) and np.sum(digit_img[-5:, :]) > 0:
                result += "."
                continue
                
            val = self.recognize_digit(digit_img)
            result += val
            
        return result
