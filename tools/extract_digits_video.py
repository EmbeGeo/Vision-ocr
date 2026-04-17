import os
import cv2
import sys
import uuid
from ultralytics import YOLO

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr.recognizer import DigitRecognizer

import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

def main():
    model_path = '../models/best.pt' if os.path.exists('../models/best.pt') else 'models/best.pt'
    video_path = '../data/samples/video_compat.mp4' if os.path.exists('../data/samples/video_compat.mp4') else 'data/samples/video_compat.mp4'
    
    # 만약 호환 비디오가 없다면 원본 비디오로 시도
    if not os.path.exists(video_path):
        video_path = video_path.replace('_compat', '')
        
    out_dir = '../data/cnn_dataset/unclassified/' if os.path.exists('../data/samples/') else 'data/cnn_dataset/unclassified/'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"==================================================")
    print(f"비디오에서 숫자 추출 시작: {video_path}")
    print(f"==================================================")
    
    model = YOLO(model_path)
    recognizer = DigitRecognizer(debug=False)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 비디오 파일을 열 수 없습니다 ({video_path})")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    # 숫자가 변하는 여유를 주기 위해 1.5초마다 1프레임씩만 추출 (중복 노가다 방지)
    frame_interval = int(fps * 1.5) 
    
    frame_idx = 0
    extracted_count = 0
    t_val = 140
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            print(f"프레임 {frame_idx} 분석 중...")
            
            # 여기서 바운딩 박스를 검출
            results = model.predict(frame, conf=0.40, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                if crop is None or crop.size == 0:
                    continue
                
                # 기존 로직으로 한 글자씩 분할
                thresh = recognizer.get_binary_image(crop, t_val)
                zones = recognizer.split_digits(thresh)
                
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                for (sx, ex) in zones:
                    if (ex - sx) < 4: continue
                    
                    digit_img = gray_crop[:, sx:ex]
                    if digit_img.shape[0] < 5 or digit_img.shape[1] < 3: continue
                    
                    uid = uuid.uuid4().hex[:6]
                    out_path = os.path.join(out_dir, f"vid_f{frame_idx}_{uid}.png")
                    cv2.imwrite(out_path, digit_img)
                    extracted_count += 1
                    
        frame_idx += 1
        
        # 사람이 일일이 수동 분류해야 하므로 너무 많아지면(200장 이상) 중단시킴
        if extracted_count > 200:
            print("\n[안내] 분류 작업의 피로도를 줄이기 위해 200장에서 추출을 조기 중단합니다.")
            break

    cap.release()
    print(f"\n[비디오 추출 완료]")
    print(f"총 {extracted_count}개의 새로운 숫자 이미지가 {os.path.abspath(out_dir)} 경로에 저장되었습니다!")

if __name__ == '__main__':
    main()
