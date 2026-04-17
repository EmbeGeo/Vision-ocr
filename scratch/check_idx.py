import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
original_load = __import__('torch').load
__import__('torch').load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('data/samples/Full1.jpeg', conf=0.40)
for idx, box in enumerate(results[0].boxes):
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    print(f"[{idx+1}] {cls_name}")
