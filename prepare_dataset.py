import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
import random
import shutil
import zipfile

# ==================================================================
# 설정 (사용자 조절 가능)
# ==================================================================
OUTPUT_DIR = 'data/dataset'
BASE_IMAGE = 'data/samples/Full1.png'
BASE_XML = 'data/samples/Full1.xml'
NUM_AUGMENTED = 500  # 생성할 총 이미지 수 (지능형 증강 적용)
TRAIN_RATIO = 0.8    # 학습 데이터 비율
# ==================================================================

CLASSES = [
    'pol1_press', 'pol2_press', 'pol1_pump_speed', 'pol2_pump_speed',
    'pol1_temp_pv', 'pol1_temp_sv', 'pol2_temp_pv', 'pol2_temp_sv',
    'hot_water_temp_pv', 'hot_water_temp_sv', 'iso_temp_pv', 'iso_temp_sv',
    'iso_pump_speed', 'iso_press'
]

def voc_to_yolo(box, img_w, img_h):
    xmin, ymin, xmax, ymax = box
    x = (xmin + xmax) / 2.0 / img_w
    y = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return (x, y, w, h)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASSES: continue
        cls_id = CLASSES.index(name)
        xmlbox = obj.find('bndbox')
        box = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
               float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
        boxes.append([cls_id, box])
    return boxes, w, h

def augment_image_and_boxes(img, boxes, w, h):
    """이미지와 박스를 동시에 변형 (회전, 노이즈, 밝기 등)"""
    aug_img = img.copy()
    new_boxes = [b[:] for b in boxes] # Deep copy
    
    # 1. 기하학적 변형 (미세 회전 및 스케일)
    angle = random.uniform(-5, 5) # -5 ~ 5도 회전
    scale = random.uniform(0.9, 1.1) # 0.9 ~ 1.1배 크기 조절
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 박스 좌표 회전 및 변환
    updated_boxes = []
    for cls_id, box in new_boxes:
        # 박스 네 모서리 점 생성
        pts = np.array([
            [box[0], box[1]], [box[2], box[1]],
            [box[0], box[3]], [box[2], box[3]]
        ])
        # 아핀 변환 적용
        ones = np.ones(shape=(len(pts), 1))
        pts_ones = np.hstack([pts, ones])
        trans_pts = M.dot(pts_ones.T).T
        
        # 새로운 bounding box 계산
        nx1 = max(0, np.min(trans_pts[:,0]))
        ny1 = max(0, np.min(trans_pts[:,1]))
        nx2 = min(w, np.max(trans_pts[:,0]))
        ny2 = min(h, np.max(trans_pts[:,1]))
        
        # YOLO 형식으로 변환
        yolo_box = voc_to_yolo([nx1, ny1, nx2, ny2], w, h)
        updated_boxes.append((cls_id, yolo_box))
    
    # 2. 색상/밝기 변형
    alpha = random.uniform(0.6, 1.4) # 대비
    beta = random.randint(-40, 40)   # 밝기
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
    
    # 3. 노이즈 및 블러
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(1, 10), aug_img.shape).astype(np.uint8)
        aug_img = cv2.add(aug_img, noise)
    
    if random.random() > 0.8:
        k = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)
        
    return aug_img, updated_boxes

def prepare():
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        path = os.path.join(OUTPUT_DIR, sub)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    img = cv2.imread(BASE_IMAGE)
    boxes, w, h = parse_xml(BASE_XML)
    
    print(f"지능형 데이터 생성 시작: 총 {NUM_AUGMENTED}장...")
    
    for i in range(NUM_AUGMENTED):
        # 이미지와 박스 동시 증강
        aug_img, updated_boxes = augment_image_and_boxes(img, boxes, w, h)
        
        is_val = random.random() > TRAIN_RATIO
        split = 'val' if is_val else 'train'
        file_name = f"Full1_aug_{i:03d}"
        
        # 이미지 저장
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', split, file_name + '.png'), aug_img)
        
        # 라벨 저장
        with open(os.path.join(OUTPUT_DIR, 'labels', split, file_name + '.txt'), 'w') as f:
            for cls_id, yolo_box in updated_boxes:
                f.write(f"{cls_id} {' '.join(map(str, yolo_box))}\n")
        
        if (i+1) % 100 == 0:
            print(f"진행 중... {i+1}/{NUM_AUGMENTED}")

    print("500장의 지능형 데이터 생성이 완료되었습니다.")

    print("dataset.zip 압축 중...")
    if os.path.exists('dataset.zip'): os.remove('dataset.zip')
    with zipfile.ZipFile('dataset.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file == '.DS_Store': continue
                rel_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                zipf.write(rel_path)
    print("dataset.zip 생성이 완료되었습니다.")

if __name__ == "__main__":
    prepare()
