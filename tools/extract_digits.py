import os
import cv2
import sys
import uuid
from ultralytics import YOLO

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr.recognizer import DigitRecognizer

# PyTorch 2.6+ 보안 설정 해결
import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

def main():
    model_path = '../models/best.pt' if os.path.exists('../models/best.pt') else 'models/best.pt'
    samples_dir = '../data/samples/' if os.path.exists('../data/samples/') else 'data/samples/'
    out_dir = '../data/cnn_dataset/unclassified/' if os.path.exists('../data/samples/') else 'data/cnn_dataset/unclassified/'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    # 기존 배경/글자 분리 로직을 활용하기 위해 임시로 Recognizer 호출
    recognizer = DigitRecognizer(debug=False)
    
    image_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No images found in data/samples/ directory.")
        return

    extracted_count = 0
    t_val = 140 # 기본 분할용 임계값
    
    for filename in image_files:
        img_path = os.path.join(samples_dir, filename)
        print(f"Processing {img_path}...")
        
        results = model.predict(img_path, conf=0.40, verbose=False)
        orig_img = results[0].orig_img
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = orig_img[y1:y2, x1:x2]
            
            if crop is None or crop.size == 0:
                continue

            # 이진화를 통해 숫자가 있는 구역을 세그멘테이션 (기존 수직 투영법 로직 사용)
            thresh = recognizer.get_binary_image(crop, t_val)
            zones = recognizer.split_digits(thresh)
            
            # 잘라낸 개별 숫자를 추출하여 저장 (회색조로 저장하는 것이 CNN에 더 유리할 수 있음)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            for (sx, ex) in zones:
                w_segment = ex - sx
                
                # 가로폭이 매우 작은 노이즈나 소수점은 분리해서 관리 (일단 스킵하거나 별도 저장)
                if w_segment < 4:
                    continue
                
                digit_img = gray_crop[:, sx:ex]
                
                # 이미지 크기가 너무 작으면 패스
                if digit_img.shape[0] < 5 or digit_img.shape[1] < 3:
                    continue
                
                uid = uuid.uuid4().hex[:8]
                out_path = os.path.join(out_dir, f"digit_{uid}.png")
                cv2.imwrite(out_path, digit_img)
                extracted_count += 1

    print(f"\n[Extraction Complete]")
    print(f"Saved {extracted_count} individual digit images to: {os.path.abspath(out_dir)}")
    print("Action Required: Please navigate to this folder and organize images into subfolders (0, 1, 2... 9, minus).")

if __name__ == '__main__':
    main()
