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
Python 3.11.5
PyTorch 2.0+
CUDA 12.6+ (for GPU acceleration)
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


### Pretrained weight
https://drive.google.com/file/d/1rVENJ2n_u-Djf55bJAzhLqU8gPlPpEBQ/view?usp=sharing

### GPU Training
main/train.py \
  --config configs/yolov8_radpose.yaml \
  --data data/coco-pose.yaml \
  --epochs 100 \
  --batch-size 16 \
  --weights yolov8s-radpose.pt

## 📝 Citation

**Note from the Authors:**  
This code repository is directly related to the manuscript titled *"Enhancing Industrial Human Pose Estimation through Hybrid Transformer-CNN Architecture"* currently submitted to **The Visual Computer** journal. If you use this code, please consider citing our work.

```bibtex
@article{yolov8radpose2025,
  title={Enhancing Industrial Human Pose Estimation through Hybrid Transformer-CNN Architecture},
  author={Muhammad Rashid, Junfeng Wang and, Sulman Ahmed},
  journal={The Visual Computer},
  year={2025},
  note={Submitted},
  url={https://github.com/rashidbhatti4399/YOLOv8-RADPose}
}
