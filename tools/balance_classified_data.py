import cv2
import numpy as np
import os
import random
import uuid
import shutil

# --- 7세그먼트 그리기 함수 정의 ---
def draw_segments(segments):
    w, h = 24, 40
    img = np.zeros((h, w), dtype=np.uint8)
    color = 255
    t = 4 # 두께
    
    # A (Top)
    if 'A' in segments: cv2.line(img, (5, 5), (w-5, 5), color, t)
    # B (Right Top)
    if 'B' in segments: cv2.line(img, (w-5, 5), (w-6, h//2 - 1), color, t)
    # C (Right Bottom)
    if 'C' in segments: cv2.line(img, (w-6, h//2 + 1), (w-7, h-5), color, t)
    # D (Bottom)
    if 'D' in segments: cv2.line(img, (w-7, h-5), (4, h-5), color, t)
    # E (Left Bottom)
    if 'E' in segments: cv2.line(img, (4, h-5), (5, h//2 + 1), color, t)
    # F (Left Top)
    if 'F' in segments: cv2.line(img, (5, h//2 - 1), (6, 5), color, t)
    # G (Center)
    if 'G' in segments: cv2.line(img, (6, h//2), (w-6, h//2), color, t)
    
    return img

DIGIT_PATTERNS = {
    '0': 'ABCDEF',
    '1': 'BC',
    '2': 'ABGED',
    '3': 'ABGCD',
    '4': 'FGBC',
    '5': 'AFGCD',
    '6': 'AFGECD', # 혹은 AFGCD (밑바닥 포함)
    '7': 'ABC',
    '8': 'ABCDEFG',
    '9': 'ABCDFG',
    'No': '' # No는 빈 이미지(혹은 극미세 잔상)
}

def apply_random_transforms(img):
    # 1. 밝기 및 대비 랜덤
    img = (img * random.uniform(0.5, 1.0)).astype(np.uint8)
    # 2. 블러
    if random.random() > 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # 3. 배경 노이즈 (매우 중요)
    bg_noise = np.random.randint(10, 60, img.shape, dtype=np.uint8)
    img = cv2.addWeighted(img, 1.0, bg_noise, 0.4, 0)
    # 4. 기하학적 변형
    scale = random.uniform(0.8, 1.1)
    M = cv2.getRotationMatrix2D((12, 20), random.uniform(-5, 5), scale)
    shear_x = random.uniform(-0.1, 0.1)
    M[0, 1] += shear_x
    img = cv2.warpAffine(img, M, (24, 40), borderMode=cv2.BORDER_REPLICATE)
    return img

def main():
    base_dir = 'data/cnn_dataset/classified'
    target_count = 30
    
    for cls in DIGIT_PATTERNS.keys():
        cls_path = os.path.join(base_dir, cls)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)
            
        current_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg'))]
        current_count = len(current_files)
        
        print(f"Class '{cls}': Current {current_count} files.")
        
        if current_count > target_count:
            # 1. 초과분 삭제 (랜덤 선택)
            to_remove = random.sample(current_files, current_count - target_count)
            for f in to_remove:
                os.remove(os.path.join(cls_path, f))
            print(f"  -> Trimmed {len(to_remove)} files. (Now {target_count})")
            
        elif current_count < target_count:
            # 2. 부족분 생성
            num_to_create = target_count - current_count
            pattern = DIGIT_PATTERNS[cls]
            for i in range(num_to_create):
                img = draw_segments(pattern)
                img = apply_random_transforms(img)
                uid = uuid.uuid4().hex[:6]
                cv2.imwrite(os.path.join(cls_path, f"balanced_synth_{uid}.png"), img)
            print(f"  -> Generated {num_to_create} synthetic files. (Now {target_count})")
        else:
            print(f"  -> Already balanced.")

if __name__ == '__main__':
    main()
