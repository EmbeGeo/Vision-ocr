import cv2
import numpy as np
import os
import random

def draw_segment_7():
    """산업용 타이포그래피 느낌의 7 세그먼트 '7' (A, B, C 구역 점등) 생성"""
    # 원본 이미지의 평균적인 크기에 맞게 생성
    w, h = 24, 40
    img = np.zeros((h, w), dtype=np.uint8)
    
    color = 255
    # 세그먼트 굵기 좀 더 두껍게
    thickness = 4
    
    # 상단 (A) - 약간 기울임(Shear)이 있는 것이 보통이므로 좌표를 조금 줍니다.
    cv2.line(img, (6, 5), (w-3, 5), color, thickness)
    # 우상단 (B)
    cv2.line(img, (w-4, 5), (w-5, h//2 - 1), color, thickness)
    # 우하단 (C)
    cv2.line(img, (w-5, h//2 + 1), (w-6, h-5), color, thickness)
    
    return img

def main():
    base_dir = '../data/cnn_dataset/classified/7' if os.path.exists('../data/cnn_dataset') else 'data/cnn_dataset/classified/7'
    os.makedirs(base_dir, exist_ok=True)
    
    count = 0
    # 기본 형태를 바탕으로 블러, 해상도 조절 등을 섞어 15가지의 변형을 만듭니다.
    for i in range(15):
        img = draw_segment_7()
        
        # 1. 픽셀 밝기 조절 (디스플레이 빛 번짐 느낌)
        img = (img * random.uniform(0.6, 1.0)).astype(np.uint8)
        
        # 2. 약간의 가우시안 블러
        if random.random() > 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
        # 3. 바탕 노이즈 (조명 반사나 어두운 회색조 배경 흉내)
        bg_noise = np.random.randint(10, 60, img.shape, dtype=np.uint8)
        img = cv2.addWeighted(img, 1.0, bg_noise, 0.4, 0)
        
        # 4. 약간 확대/축소 등 (기하학 변형)
        scale = random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((12, 20), 0, scale) # 회전은 고정, 크기만
        img = cv2.warpAffine(img, M, (24, 40))
        
        cv2.imwrite(os.path.join(base_dir, f"synth_7_{i}.png"), img)
        count += 1
        
    print(f"가상의 '7' 데이터 {count}장을 성공적으로 생성하여 {base_dir} 폴더에 넣었습니다.")

if __name__ == '__main__':
    main()
