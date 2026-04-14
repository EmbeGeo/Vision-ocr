import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# 추론 시에 모델 구조가 필요하므로 tools/train_cnn.py에서 정의한 것과 동일한 클래스 선언을 유지합니다.
class OcrDigitCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(OcrDigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CnnDigitRecognizer:
    """
    CNN 모델 기반 OCR 추론 엔진
    기존 DigitRecognizer (Zonal Analysis)를 완벽히 대체(Drop-in replacement)하도록 설계되었습니다.
    """
    def __init__(self, model_path='models/cnn_digit_best.pth', debug=False):
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(base_dir, model_path)
            
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        if not os.path.exists(full_model_path):
            print(f"[Warning] CNN Weights missing at {full_model_path}.")
            print("Please run data extraction and training scripts first.")
            self.model = None
            return
            
        # 체크포인트 로드 (학습된 모델 파라미터 + 데이터셋 클래스 매핑 정보)
        checkpoint = torch.load(full_model_path, map_location=self.device)
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        num_classes = len(self.class_to_idx)
        
        self.model = OcrDigitCNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_binary_image(self, crop, thresh_val):
        """숫자 분할 기준을 찾기 위한 이진화"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        return thresh

    def split_digits(self, thresh):
        """수직 투영(Vertical Projection)을 이용한 공백 기반 숫자 단위 분할"""
        h, w = thresh.shape
        projection = np.sum(thresh, axis=0)
        
        digit_zones = []
        start = None
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

    def predict_digit(self, digit_img):
        """단일 숫자 이미지를 CNN에 통과시켜 클래스 예측 반환"""
        if self.model is None:
            return "?"
            
        # OpenCV numpy array -> PIL Image
        if len(digit_img.shape) == 2:
            pil_img = Image.fromarray(digit_img).convert('L')
        else:
            pil_img = Image.fromarray(cv2.cvtColor(digit_img, cv2.COLOR_BGR2RGB))
            
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        if confidence.item() < 0.5:
            return "?"
            
        class_idx = predicted.item()
        return self.idx_to_class[class_idx]
        
    def read(self, crop, var_name="unknown"):
        """입력된 크롭 이미지(바운딩 박스)에서 전체 문자열 반환"""
        if crop is None or crop.size == 0: return ""
        
        # 다중 임계값 대신 고정 임계값으로 빠른 분리 처리
        t_val = 140
        thresh = self.get_binary_image(crop, t_val)
        
        zones = self.split_digits(thresh)
        
        result_str = ""
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        for (sx, ex) in zones:
            digit_crop = gray_crop[:, sx:ex]
            
            # 노이즈 크기 필터링
            if digit_crop.shape[1] < 4 or digit_crop.shape[0] < 5:
                continue
                
            pred = self.predict_digit(digit_crop)
            
            if pred == "?":
                pass
            elif pred.lower() == 'no':
                pass
            elif pred == 'minus':
                result_str += '-'
            elif pred == 'dot':
                result_str += '.'
            else:
                result_str += str(pred)
                
        return result_str
