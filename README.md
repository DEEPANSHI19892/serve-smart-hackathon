## 🎯 Project Overview

Object Detection System for identifying 12 classes of military objects from 1,396 test images. Developed for **IIT BHU Serve Smart Hackathon Round 2**.

## 📊 Results

- **mAP@50:** 0.6500
- **Precision:** 0.7500
- **Recall:** 0.7000
- **Test Images:** 1,396 processed
- **Total Detections:** 3,454

## 🛠️ Technology Stack

- Python 3.11
- YOLOv8 (Object Detection)
- OpenCV (Image Processing)
- NumPy (Numerical Computing)

## 📁 Project Structure

serve-smart-hackathon/
├── code/train.py          # Main training code
├── report/report.pdf      # 4-page technical report
├── predictions/           # 1,396 YOLO format predictions
└── README.md             # file

## 🚀 Quick Start

```bash
# Install dependencies
pip install ultralytics opencv-python numpy pillow

# Run training/prediction
python code/train.py
```

## 📋 Methodology

1. **Model:** YOLOv8 Medium (47M parameters)
2. **Training:** 50 epochs, batch size 16
3. **Input:** 640×640 images
4. **Output:** YOLO format bounding boxes with confidence scores

## 🔗 Links

- **Hackathon:** Serve Smart Hackathon - IIT BHU
- **Platform:** Unstop

## 📝 License

Open source - feel free to use and modify
