from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the model
model = YOLO("/Users/namyunhun/mm/best (2).pt")
results = model.predict("/Users/namyunhun/Downloads/개/몸/KakaoTalk_Photo_2023-12-02-20-09-33.png")

# Extract class ID and confidence score
class_id = int(results[0].boxes.data[0, 5])
confidence_score = results[0].boxes.data[0, 4]

# Get the class name from the 'names' dictionary
class_name = results[0].names[class_id]

# Print the results
print(f"Class Name: {class_name}")
print(f"Confidence Score: {confidence_score:.4f}")
