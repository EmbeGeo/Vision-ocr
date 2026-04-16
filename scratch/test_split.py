import sys
import os
import cv2
import numpy as np

# Load model to use its crop logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from ultralytics import YOLO

model = YOLO('../models/best.pt')
results = model.predict('../data/samples/Full1.jpeg', conf=0.40)

crops = {}
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    crop = results[0].orig_img[y1:y2, x1:x2]
    crops[cls_name] = crop

out_dir = 'debug_crops'
os.makedirs(out_dir, exist_ok=True)

# Function to test splitting
def try_split(crop_img, name):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    # also try otsu
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite(f'{out_dir}/{name}_crop.png', crop_img)
    cv2.imwrite(f'{out_dir}/{name}_th140.png', thresh)
    cv2.imwrite(f'{out_dir}/{name}_thOtsu.png', thresh_otsu)
    
    # Check vertical projection of Otsu
    h, w = thresh_otsu.shape
    proj = np.sum(thresh_otsu, axis=0)
    print(f"{name}: width={w}, proj_zeros(otsu)={np.sum(proj==0)}")
    
    proj140 = np.sum(thresh, axis=0)
    print(f"{name}: width={w}, proj_zeros(140)={np.sum(proj140==0)}")

for name in ['pol2_press', 'pol2_temp_sv', 'hot_water_temp_pv', 'hot_water_temp_sv', 'iso_press', 'iso_temp_pv', 'iso_temp_sv']:
    if name in crops:
        try_split(crops[name], name)
