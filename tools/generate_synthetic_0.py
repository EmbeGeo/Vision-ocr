import cv2
import numpy as np
import os
import random
import uuid

def draw_segment_0():
    """산업용 7세그먼트 '0' (A, B, C, D, E, F 점등) 생성"""
    w, h = 24, 40
    img = np.zeros((h, w), dtype=np.uint8)
    
    color = 255
    thickness = 4
    
    # A (Top)
    cv2.line(img, (5, 5), (w-5, 5), color, thickness)
    # B (Right Top)
    cv2.line(img, (w-5, 5), (w-6, h//2 - 1), color, thickness)
    # C (Right Bottom)
    cv2.line(img, (w-6, h//2 + 1), (w-7, h-5), color, thickness)
    # D (Bottom)
    cv2.line(img, (w-7, h-5), (4, h-5), color, thickness)
    # E (Left Bottom)
    cv2.line(img, (4, h-5), (5, h//2 + 1), color, thickness)
    # F (Left Top)
    cv2.line(img, (5, h//2 - 1), (6, 5), color, thickness)
    
    return img

def main():
    out_dir = 'data/cnn_dataset/unclassified'
    os.makedirs(out_dir, exist_ok=True)
    
    count = 0
    for i in range(50):
        img = draw_segment_0()
        
        # 1. 밝기 및 대비 랜덤
        img = (img * random.uniform(0.6, 1.0)).astype(np.uint8)
        
        # 2. 가우시안 블러 (빛 번짐)
        if random.random() > 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
            
        # 3. 배경 노이즈
        bg_noise = np.random.randint(10, 60, img.shape, dtype=np.uint8)
        img = cv2.addWeighted(img, 1.0, bg_noise, 0.4, 0)
        
        # 4. 기하학적 변형 (회전, 스케일, 기울임)
        scale = random.uniform(0.8, 1.1)
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((12, 20), angle, scale)
        
        # Shear (기울임 추가)
        shear_x = random.uniform(-0.1, 0.1)
        M[0, 1] += shear_x
        
        img = cv2.warpAffine(img, M, (24, 40), borderMode=cv2.BORDER_REPLICATE)
        
        uid = uuid.uuid4().hex[:6]
        cv2.imwrite(os.path.join(out_dir, f"synth_0_{uid}.png"), img)
        count += 1
        
    print(f"가상의 '0' 데이터 {count}장을 성공적으로 생성하여 {out_dir} 폴더에 넣었습니다.")

if __name__ == '__main__':
    main()
