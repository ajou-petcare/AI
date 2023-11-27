#!/usr/bin/env python
# coding: utf-8

# ##### image_path: jpg 디렉토리, json_path: json 디렉토리

# In[1]:


image_path = '/home/mlmlab08/dog_body/Data/image'
json_path = '/home/mlmlab08/dog_body/Data/label'


# ## 1. yolo model github로부터 clone

# In[3]:


#git clone https://github.com/ultralytics/yolov5.git


# In[ ]:


#cd yolov5
#pip install -r requirements.txt


# In[2]:


from PIL import Image
import os
import json


# In[3]:


image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')])
json_files = sorted([os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')])


# In[7]:


from PIL import Image

def convert_to_yolo_format(json_data, img_width, img_height):



    label_data = json_data['annotations']
    yolo_data = []
    class_id = json_data['metadata']['physical']['BCS'] - 1

    # Bounding box 좌표 추출
    x_min, y_min, x_max, y_max = [float(coord) for coord in label_data['label']['points'][0]] + [float(coord) for coord in label_data['label']['points'][1]]

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


# In[9]:


for img_file, json_file in zip(image_files, json_files):
    # 이미지 크기를 읽음
    img = Image.open(img_file)
    img_width, img_height = img.size

    # JSON 파일을 읽음
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # JSON 정보를 YOLO 형식으로 변환
    yolo_format_data = convert_to_yolo_format(json_data, img_width, img_height)

    # YOLO 형식의 데이터를 파일로 저장
    label_file = os.path.join('/home/mlmlab08/dog_body/Data/yolo-label', os.path.basename(img_file).replace('.jpg', '.txt'))
    with open(label_file, 'w') as file:
        file.write(yolo_format_data)


# In[12]:


import os
import shutil
import random

# 이미지와 라벨이 저장된 디렉토리
image_data_dir = '/home/mlmlab08/dog_body/Data/image'
label_data_dir = '/home/mlmlab08/dog_body/Data/yolo-label'

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
os.makedirs("/home/mlmlab08/dog_body/dataset/train/images", exist_ok=True)
os.makedirs("/home/mlmlab08/dog_body/dataset/train/labels", exist_ok=True)
os.makedirs("/home/mlmlab08/dog_body/dataset/val/images", exist_ok=True)
os.makedirs("/home/mlmlab08/dog_body/dataset/val/labels", exist_ok=True)

# 분할된 파일들을 적절한 디렉토리로 복사
for img, label in zip(train_images, train_labels):
    shutil.copy(os.path.join(image_data_dir, img), os.path.join("/home/mlmlab08/dog_body/dataset/train/images", img))
    shutil.copy(os.path.join(label_data_dir, label), os.path.join("/home/mlmlab08/dog_body/dataset/train/labels", label))

for img, label in zip(val_images, val_labels):
    shutil.copy(os.path.join(image_data_dir, img), os.path.join("/home/mlmlab08/dog_body/dataset/val/images", img))
    shutil.copy(os.path.join(label_data_dir, label), os.path.join("/home/mlmlab08/dog_body/dataset/val/labels", label))


# In[13]:


#os.chdir("./yolov5")




from ultralytics import YOLO
model = YOLO('yolov5n6.yaml')
model = YOLO('yolov5s.pt')


# In[16]:


train_results = model.train(
    data='/home/mlmlab08/dog_body/yolov5/data/dog_body_bcs_yolo_config.yaml', 
    epochs=2000,
    patience=3,
    mixup=0.1,
    project='yolov5-dogbody',
    device=2
)


# In[ ]:


valid_results = model.val()

