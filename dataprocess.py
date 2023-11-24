## 데이터를 전처리하기 위한 파일
### 이 파일을 통해 반려동물의 데이터를 전처리할 수  있도록 진행한다.
from PIL import Image
import os
import json
import random
import shutil

image_path = "/dataset/imagedata"
json_path = "/dataset/labeldata"

# json파일을 yolo형식으로 바꿔주는 메소드
def convert_to_yolo_format(json_data, img_width, img_height):
    annotations = json_data['annotations']
    yolo_data = []

    for ann in annotations:
        class_id = int(ann['class']) - 1  # Assuming class IDs start from 1 in JSON
        x_center = (ann['box'][0] + ann['box'][2]) / 2.0
        y_center = (ann['box'][1] + ann['box'][3]) / 2.0
        width = ann['box'][2] - ann['box'][0]
        height = ann['box'][3] - ann['box'][1]

            # Normalize to image dimensions
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return "\n".join(yolo_data)

def divide_data():
    image_data_dir = "/dataset/refinedata/images"
    label_data_dir = "/dataset/refinedata/labels"
    # 파일 리스트를 가져옴
    image_files = sorted([f for f in os.listdir(image_data_dir) if f.endswith('.jpg')])
    label_files = sorted([f.replace('.jpg', '.txt') for f in image_files])

    # 파일 리스트를 무작위로 섞음
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    image_files, label_files = zip(*combined)

    # 분할 비율 설정 8:2
    split_ratio = 0.8
    split_idx = int(len(image_files) * split_ratio)

    # 파일 분할
    train_images = image_files[:split_idx]
    train_labels = label_files[:split_idx]
    val_images = image_files[split_idx:]
    val_labels = label_files[split_idx:]

    # 분할된 파일들을 적절한 디렉토리로 이동
    for img, label in zip(train_images, train_labels):
        shutil.move(os.path.join(image_data_dir, img), os.path.join("/dataset/refinedata/images/train", img))
        shutil.move(os.path.join(label_data_dir, label), os.path.join("/dataset/refinedata/labels/train", label))

    for img, label in zip(val_images, val_labels):
        shutil.move(os.path.join(image_data_dir, img), os.path.join("/dataset/refinedata/images/val", img))
        shutil.move(os.path.join(label_data_dir, label), os.path.join("/dataset/refinedata/labels/val", label))


def main():
    image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')])
    json_files = sorted([os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')])

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
        label_file = os.path.join("/dataset/yolo-label", os.path.basename(img_file).replace('.jpg', '.txt'))
        with open(label_file, 'w') as file:
            file.write(yolo_format_data)
            
    # 데이터셋 분할
    divide_data()