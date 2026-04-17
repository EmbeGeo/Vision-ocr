import os
import cv2
import glob
import random
import shutil
import numpy as np

def apply_augmentations(img):
    """
    OpenCV를 사용한 기본 데이터 증강 적용
    가벼운 변형을 랜덤하게 적용하여 오버피팅을 방지합니다.
    """
    aug_fns = [
        # 밝기 및 대비 조정
        lambda x: cv2.convertScaleAbs(x, alpha=random.uniform(0.8, 1.2), beta=random.randint(-15, 15)),
        # 아주 가벼운 가우시안 블러
        lambda x: cv2.GaussianBlur(x, (3, 3), 0) if random.random() > 0.5 else x,
        # 노이즈 추가
        add_noise,
        # 픽셀 단위로 상하좌우 이동 처리
        translate_image
    ]
    
    out = img.copy()
    for fn in aug_fns:
        if random.random() > 0.4:  # 60% 확률로 각 효과 적용
            out = fn(out)
    return out

def add_noise(img):
    row, col = img.shape
    # 가벼운 가우시안 노이즈
    gauss = np.random.normal(0, 4, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def translate_image(img):
    rows, cols = img.shape
    # 숫자 이미지 특성상 중심에서 너무 벗어나지 않게 -2~2 픽셀만 이동
    tx = random.randint(-2, 2)
    ty = random.randint(-2, 2)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # Replicate border mode로 경계를 자연스럽게 연장
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

def main():
    base_dir = '../data/cnn_dataset' if os.path.exists('../data/cnn_dataset') else 'data/cnn_dataset'
    classified_dir = os.path.join(base_dir, 'classified')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    if not os.path.exists(classified_dir) or not os.listdir(classified_dir):
        print("=====================================================")
        print(f"Error: {classified_dir} 폴더가 없거나 비어 있습니다.")
        print("이전 단계에서 추출된 숫자들을 'classified' 내부의")
        print("각 클래스 폴더(0, 1, 2... 9)로 이동시킨 후 실행해주세요.")
        print("=====================================================")
        return

    # 증강 목표 설정 (본 학습에서는 데이터 퀄리티에 따라 조정 필요)
    TARGET_COUNT_PER_CLASS = 1500
    VAL_SPLIT = 0.2
    
    # 기존 데이터 초기화 (완전 교체)
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    classes = [d for d in os.listdir(classified_dir) if os.path.isdir(os.path.join(classified_dir, d))]
    
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls))
        os.makedirs(os.path.join(val_dir, cls))
        
        cls_path = os.path.join(classified_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"Warning: '{cls}' 클래스 폴더가 비어 있어 스킵됩니다.")
            continue
            
        print(f"Augmenting class '{cls}' (Total source images: {len(images)})...")
        
        # Validation 데이터 생성
        val_count = max(1, int(TARGET_COUNT_PER_CLASS * VAL_SPLIT))
        train_count = TARGET_COUNT_PER_CLASS - val_count
        
        for i in range(val_count):
            src_img_path = os.path.join(cls_path, random.choice(images))
            img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            
            # 리사이즈 기준 고정 (ex: 28x28 또는 32x32)
            # 여기서는 32x32를 기준으로 잡아둡니다. 학습 시 편리함
            img = cv2.resize(img, (32, 32))
            aug_img = apply_augmentations(img) 
            cv2.imwrite(os.path.join(val_dir, cls, f"val_{i}.png"), aug_img)
            
        # Train 데이터 생성
        for i in range(train_count):
            src_img_path = os.path.join(cls_path, random.choice(images))
            img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            
            img = cv2.resize(img, (32, 32))
            aug_img = apply_augmentations(img)
            cv2.imwrite(os.path.join(train_dir, cls, f"train_{i}.png"), aug_img)

    print("\n[Augmentation Complete]")
    print(f"Target count per class : {TARGET_COUNT_PER_CLASS}")
    print(f"Train dataset location : {os.path.abspath(train_dir)}")
    print(f"Val dataset location   : {os.path.abspath(val_dir)}")

if __name__ == '__main__':
    main()
