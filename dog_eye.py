from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


# Load the model
model = YOLO("/Users/namyunhun/mm/AI/yolov5-dogeye/train/weights/best.pt")
results = model.predict("/Users/namyunhun/Downloads/개/눈/스크린샷 2023-12-13 오전 9.47.16.png")

# Check if detections are present
if len(results[0].boxes.data) > 0:
    # Extract class ID, confidence score, and class name for each detection
    for i in range(len(results[0].boxes.data)):
        class_id = int(results[0].boxes.data[i, 5])
        confidence_score = results[0].boxes.data[i, 4]
        class_name = results[0].names[class_id]

        print(f"Detection {i+1}: Class Name: {class_name}, Confidence Score: {confidence_score:.4f}")
else:
    print("No detections found in the image.")
