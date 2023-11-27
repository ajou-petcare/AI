#!/usr/bin/env python
# coding: utf-8

# ##### image_path: jpg 디렉토리, json_path: json 디렉토리

# In[2]:


image_path = '/home/mlmlab08/dogeye/Data/image'
json_path = '/home/mlmlab08/dogeye/Data/label'


# ## 1. yolo model github로부터 clone

# In[3]:


#git clone https://github.com/ultralytics/yolov5.git


# In[ ]:


#cd yolov5
#pip install -r requirements.txt


# In[4]:


from PIL import Image
import os
import json


# In[5]:


image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')])
json_files = sorted([os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')])


# In[16]:


def convert_to_yolo_format(json_data):
    # 이미지 너비와 높이 추출
    img_width, img_height = [float(dim) for dim in json_data['images']['meta']['width_height']]
    label_data = json_data['label']
    yolo_data = []
    class_dict = {
        ('결막염', '무'): 0,
        ('결막염', '유'): 1,
        ('백내장', '무'): 0,
        ('백내장', '성숙'): 2,
        ('유루증', '유'): 3,
        ('안검염', '유'): 4,
        ('핵경화', '유'): 5,
        ('안검종양', '유'): 6,
        ('안검내반증', '유'): 7,
        ('궤양성각막질환', '상'): 8,
        ('비궤양성각막질환', '상'): 9,
        ('색소침착성각막염', '유'): 10
    }

    # JSON 파일 구조에 따라 label_data가 리스트가 아니라면 리스트로 변경
    if not isinstance(label_data, list):
        label_data = [label_data]

    for label in label_data:
        disease_name = label['label_disease_nm']
        disease_level = label['label_disease_lv_3']
        class_id = class_dict.get((disease_name, disease_level), -1)

        if class_id == -1:
            continue

        x_min, y_min, x_max, y_max = [float(coord) for coord in label['label_bbox']]

        # YOLO 포맷 값 계산
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = abs(x_max - x_min)
        height = abs(y_max - y_min)

        # 이미지 크기에 따라 정규화
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return "\n".join(yolo_data)

from PIL import Image


# In[17]:


for img_file, json_file in zip(image_files, json_files):
    # 이미지 크기를 읽음
    img = Image.open(img_file)
    img_width, img_height = img.size

    # JSON 파일을 읽음
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # JSON 정보를 YOLO 형식으로 변환
    yolo_format_data = convert_to_yolo_format(json_data)

    # YOLO 형식의 데이터를 파일로 저장
    label_file = os.path.join('/home/mlmlab08/dogeye/Data/yolo-label', os.path.basename(img_file).replace('.jpg', '.txt'))
    with open(label_file, 'w') as file:
        file.write(yolo_format_data)


# In[20]:


import os
import shutil
import random

# 이미지와 라벨이 저장된 디렉토리
image_data_dir = '/home/mlmlab08/dogeye/Data/image'
label_data_dir = '/home/mlmlab08/dogeye/Data/yolo-label'

# 파일 리스트를 가져옴
image_files = sorted([f for f in os.listdir(image_data_dir) if f.endswith('.jpg')])
label_files = sorted([f.replace('.jpg', '.txt') for f in image_files])

# 파일 리스트를 무작위로 섞음
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files, label_files = zip(*combined)

# 분할 비율 설정 (예: 80% 학습, 20% 검증)
split_ratio = 0.8
split_idx = int(len(image_files) * split_ratio)

# 파일 분할
train_images = image_files[:split_idx]
train_labels = label_files[:split_idx]
val_images = image_files[split_idx:]
val_labels = label_files[split_idx:]

# 학습 및 검증 데이터셋 디렉토리 생성
os.makedirs("/home/mlmlab08/dogeye/dataset/train/images", exist_ok=True)
os.makedirs("/home/mlmlab08/dogeye/dataset/train/labels", exist_ok=True)
os.makedirs("/home/mlmlab08/dogeye/dataset/val/images", exist_ok=True)
os.makedirs("/home/mlmlab08/dogeye/dataset/val/labels", exist_ok=True)

# 분할된 파일들을 적절한 디렉토리로 복사
for img, label in zip(train_images, train_labels):
    shutil.copy(os.path.join(image_data_dir, img), os.path.join("/home/mlmlab08/dogeye/dataset/train/images", img))
    shutil.copy(os.path.join(label_data_dir, label), os.path.join("/home/mlmlab08/dogeye/dataset/train/labels", label))

for img, label in zip(val_images, val_labels):
    shutil.copy(os.path.join(image_data_dir, img), os.path.join("/home/mlmlab08/dogeye/dataset/val/images", img))
    shutil.copy(os.path.join(label_data_dir, label), os.path.join("/home/mlmlab08/dogeye/dataset/val/labels", label))


# In[26]:


#os.chdir("./yolov5")


# In[29]:


from ultralytics import YOLO
model = YOLO('yolov5n6.yaml')
model = YOLO('yolov5s.pt')


# In[31]:


train_results = model.train(
    data='/home/mlmlab08/dogeye/yolov5/data/yolo_config.yaml', 
    epochs=2000,
    patience=3,
    mixup=0.1,
    project='yolov5-dogeye',
    device=1
)


# In[33]:


valid_results = model.val()


# In[3]:


from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the model
model = YOLO('/home/mlmlab08/dogeye/yolov5/yolov5-dogeye/train2/weights/best.pt')
results = model.predict('/home/mlmlab08/dogeye/Data/image/crop_D0_0e2efca8-60a5-11ec-8402-0a7404972c70.jpg')

# Plot the results
res_plotted = results[0].plot()

# Convert the color from BGR to RGB (fixing the missing argument error)
res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(12,12))
plt.imshow(res_plotted_rgb)
plt.show()

