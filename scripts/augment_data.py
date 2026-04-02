import cv2
import os
import numpy as np
import random

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 50
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(image):
    alpha = random.uniform(0.7, 1.3) # Contrast control (1.0-3.0)
    beta = random.uniform(-30, 30)   # Brightness control (0-100)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_blur(image):
    ksize = random.choice([(3,3), (5,5), (7,7)])
    return cv2.GaussianBlur(image, ksize, 0)

def random_affine(image):
    rows, cols = image.shape[:2]
    # random rotation -5 to +5 degrees
    angle = random.uniform(-5, 5)
    # random scale 0.9 to 1.1
    scale = random.uniform(0.9, 1.1)
    
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    
    # random translation
    tx = random.uniform(-0.05, 0.05) * cols
    ty = random.uniform(-0.05, 0.05) * rows
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

def main():
    input_images = ["tests/samples/Full1.png", "tests/samples/Full2.png"]
    output_dir = "dataset/images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_augmentations_per_image = 50
    total_generated = 0
    
    for img_path in input_images:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found.")
            continue
            
        img = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save original 
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_orig.png"), img)
        total_generated += 1
        
        print(f"Augmenting {img_path}...")
        for i in range(num_augmentations_per_image):
            aug_img = img.copy()
            
            # Apply random sequence of augmentations
            if random.random() < 0.6:
                aug_img = adjust_brightness_contrast(aug_img)
            if random.random() < 0.4:
                aug_img = add_noise(aug_img)
            if random.random() < 0.3:
                aug_img = apply_blur(aug_img)
            if random.random() < 0.7:
                aug_img = random_affine(aug_img)
                
            out_path = os.path.join(output_dir, f"{base_name}_aug_{i:03d}.png")
            cv2.imwrite(out_path, aug_img)
            total_generated += 1
            
    print(f"Done! Generated {total_generated} training images in {output_dir}")

if __name__ == "__main__":
    main()
