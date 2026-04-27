import cv2
import easyocr
import warnings
import numpy as np
import torch

class EasyOcrRecognizer:
    """
    EasyOCR 기반 인식기 (Drop-in replacement for CnnDigitRecognizer)
    상용급 딥러닝 텍스트 인식 엔진을 사용하여 높은 정확성을 보장합니다.
    """
    def __init__(self):
        # 관련 없는 시스템 경고 무시
        warnings.filterwarnings("ignore", category=UserWarning)
        
        print("[System] EasyOCR 모델 초기화 중... (최초 실행 시 모델 다운로드에 약간의 시간이 소요될 수 있습니다)")
        # 숫자 인식을 위해 영어('en') 언어 로드. GPU 가속 자동 감지
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
    def read(self, crop, var_name="unknown", box_idx=None):
        if crop is None or crop.size == 0:
            return ""
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        
        # 기본 임계값 세팅
        v_thresh = 100
        t_thresh = 0.9
        l_text = 0.6
        
        # [변수별 맞춤형 임계값 설정]
        if var_name == 'pol2_temp_pv': v_thresh = 140
        elif var_name == 'pol2_temp_sv': v_thresh = 120
        elif var_name == 'pol1_temp_sv': v_thresh = 120
        elif var_name == 'pol1_temp_pv': v_thresh = 140
        elif var_name == 'iso_temp_pv': v_thresh = 140
        elif var_name == 'iso_temp_sv': v_thresh = 150
        elif var_name == 'iso_press':
            v_thresh = 95                 
            t_thresh = 0.45                
            l_text = 0.35
            
        _, mask = cv2.threshold(v, v_thresh, 255, cv2.THRESH_BINARY)
        crop_clean = cv2.bitwise_and(crop, crop, mask=mask)
            
        # EasyOCR은 내부 CRAFT 모델로 글자를 찾기 때문에, 
        # 이미지가 작을 경우 글자 판독력이 떨어집니다.
        # 따라서 높이가 64픽셀 미만인 작은 크롭 영역은 크기를 키워줍니다.
        h, w = crop_clean.shape[:2]
        if h < 64:
            scale = 64.0 / h
            target_img = cv2.resize(crop_clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            target_img = crop_clean
            
        # 숫자 및 소수점, 마이너스 부호만 판독하도록 화이트리스트 강제 설정
        allowlist = '0123456789.-'
        
        # detail=0 옵션으로 순수 텍스트 결과만 반환받음
        results = self.reader.readtext(target_img, allowlist=allowlist, detail=0, 
                                       text_threshold=t_thresh, low_text=l_text, mag_ratio=1.5)
        
        if not results:
            return ""
            
        # 결과 결합 (혹시 띄어쓰기로 분리되어 인식된 경우 처리)
        result_str = "".join(results)
        
        # 소수점 누락에 대비한 기존 포맷 보정 로직 유지
        DECIMAL_CLASSES = {'pol1_press', 'pol2_press', 'iso_press'}
        if var_name in DECIMAL_CLASSES:
            digits_only = result_str.replace('-', '').replace('.', '')
            if '.' not in result_str:
                if len(digits_only) >= 2:
                    result_str = digits_only[:-1] + '.' + digits_only[-1]
                elif len(digits_only) == 1:
                    result_str = '0.' + digits_only
                    
        return result_str
