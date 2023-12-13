from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("/Users/namyunhun/mm/AI/yolov5-cateye/train/weights/best.pt")
results = model.predict("/Users/namyunhun/Downloads/개/고양이눈/스크린샷 2023-12-13 오전 9.41.25.png")

# Check if detections are present

class_id = int(results[0].boxes.data[0, 5])
confidence_score = results[0].boxes.data[0, 4]
class_name = results[0].names[class_id]

print(f"Class Name: {class_name}")
print(f"Confidence Score: {confidence_score:.4f}")

print("No detections found in the image.")
