# YOLOv8-RADPose
Official implementation of 'Enhancing Industrial Human Pose Estimation through Hybrid Transformer-CNN Architecture

Human pose estimation (HPE) plays a pivotal role in modern industrial systems, particularly in human-robot collaboration (HRC). Accurate and real-time HPE under challenging conditions such as occlusion, irregular body orientation, and complex lighting is essential for safety-critical decision-making. This paper introduces YOLOv8-RADPose, a hybrid Transformer-CNN architecture that integrates Resolution-Aware Dense (RAD) Transformer blocks into the YOLOv8 pipeline. The proposed architecture enhances feature extraction and global contextual reasoning, achieving significant improvements in accuracy and robustness.
# Dataset Preparation

This directory contains utilities for preparing and processing datasets for YOLOv8-RADPose.

## 📊 Supported Datasets

# COCO: https://cocodataset.org/

# CrowdPose: https://github.com/Jeff-sjtu/CrowdPose

# InHARD: http://inhard-dataset.iais.fraunhofer.de/

# Enivornment installation

Install pytorch >= v2.0.0 following official instruction https://pytorch.org/.

# Install dependencies

pip install -r requirements.txt

**Core requirements:**
Python 3.9+
PyTorch 2.0+
CUDA 11.8+ (for GPU acceleration)
Ultralytics 8.0+
OpenCV 4.7+

### 📁 Dataset Preparation
### 1. MS COCO
**Download:**
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

### GPU Training
main/train.py \
  --config configs/train_config.yaml \
  --data data/coco.yaml \
  --epochs 300 \
  --batch-size 64 \
  --weights yolov8s-pose.pt
