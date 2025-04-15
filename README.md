# **PotholeGuard: Pothole Detection and Segmentation using YOLOv8**

PotholeGuard is a deep learning-based system for automatic detection and segmentation of potholes in road images and videos. It leverages the **YOLOv8n-seg** architecture to localize potholes and generate pixel-level masks, enabling severity estimation through area analysis.

This project demonstrates how pre-trained segmentation models can be fine-tuned for custom object detection tasks using a compact, annotated dataset.

---

## **Key Features**

- **Pothole Detection & Segmentation**
  - Uses YOLOv8n-seg to detect and segment potholes from images and video frames.

- **Pixel-Level Area Calculation**
  - Measures the area (in pixels) of segmented potholes to estimate severity.

- **Custom Dataset Fine-Tuning**
  - Model fine-tuned on a domain-specific pothole dataset via transfer learning.

- **Evaluation Metrics**
  - Evaluated using standard object detection and segmentation metrics.

- **Visualizations**
  - Annotated image and video outputs showing detected potholes with area summaries.

---

## **Dataset & Preprocessing**

- **Dataset Source**
  - **Dataset Link**: [Pothole Image Segmentation Dataset](https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset)
  - **Roboflow Dataset Page**: [Pothole Segmentation YOLOv8](https://universe.roboflow.com/farzad/pothole_segmentation_yolov8/dataset/1)
  - **License**: CC BY 4.0

- **Dataset Details**
  - Training Images: 720
  - Validation Images: 60
  - Number of Classes: 1 (Pothole)
  - Image Resolution: 640x640
  - Annotation Format: YOLOv8 segmentation masks

- **Preprocessing Steps**
  - Verified dataset integrity (checked for missing and corrupt images)
  - Visualized sample images to confirm annotation accuracy

---

### **Model Architecture & Training Strategy**

- **Base Model**
  - Architecture: YOLOv8n-seg (lightweight segmentation variant)
  - Pretrained on: COCO dataset (pothole class not included)

- **Training Configuration**
  - Framework: Ultralytics YOLOv8
  - Transfer learning applied to adapt the model for pothole detection
  - Loss Functions: Bounding box loss, classification loss, mask segmentation loss
  - Training Environment: Google Colab with GPU acceleration
  - Dataset Size: Small (~780 images), optimized for efficient training

---

### **Evaluation & Visualizations**

- **Metrics Used**
  - Object Detection: Precision, Recall, mAP@0.5, mAP@0.5:0.95
  - Segmentation: IoU-based precision, recall, mAP@0.5, mAP@0.5:0.95

- **Evaluation Tools**
  - Annotated validation images with predicted masks
  - Precision-Recall curves for object detection and segmentation
  - Confusion matrix to analyze class-wise predictions
  - Side-by-side image and video frames showing pothole area summaries
  - Real-time frame-by-frame pothole detection in video clips

---

### **Technology Stack**

- **Deep Learning Framework:** YOLOv8 (Ultralytics)
- **Programming Language:** Python
- **Development Environment:** Google Colab
- **Visualization Tools:** OpenCV, Matplotlib, Seaborn
- **Data Handling:** Pandas, NumPy
- **Annotation & Labeling:** Roboflow (optional)

---

### **Project Highlights**

- Fine-tuned YOLOv8n-seg on a custom dataset for pothole detection and segmentation.
- Developed an end-to-end pipeline from preprocessing to visual inference.
- Utilized segmentation masks for pixel-level severity estimation.
- Demonstrates effective use of transfer learning for small-data object segmentation tasks.
