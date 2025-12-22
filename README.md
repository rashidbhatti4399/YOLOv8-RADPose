# YOLOv8-RADPose
Official implementation of 'Enhancing Industrial Human Pose Estimation through Hybrid Transformer-CNN Architecture

# Dataset Preparation

This directory contains utilities for preparing and processing datasets for YOLOv8-RADPose.

## 📊 Supported Datasets

The datasets used in this study are publicly available from the following sources: 
the CrowdPose dataset can be accessed at https://github.com/jeffffffli/CrowdPose, 
MS COCO dataset is available at https://cocodataset.org 
and the InHARD dataset is available at https://github.com/vhavard/InHARD?tab=readme-ov-file.

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
