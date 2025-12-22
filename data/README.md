# YOLOv8-RADPose
Official implementation of 'Enhancing Industrial Human Pose Estimation through Hybrid Transformer-CNN Architecture

# Dataset Preparation

This directory contains utilities for preparing and processing datasets for YOLOv8-RADPose.

## 📊 Supported Datasets

### 1. MS COCO
**Download:**
```bash
mkdir -p data/coco
cd data/coco

# Download images (2017 version)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

# Extract
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

**Structure:**
coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
├── val2017/
└── test2017/
