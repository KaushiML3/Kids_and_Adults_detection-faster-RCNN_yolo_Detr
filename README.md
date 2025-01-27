# Kids_and_Adults_detection

This repository contains the code and resources for developing a deep learning model to detect and classify individuals as kids or adults in images or videos.

## Features:
- Object Detection: Detects people in images with bounding boxes.
- Classification: Classifies each detected individual as either a kid or an adult.
- Transfer Learning: Utilizes pre-trained object detection models for faster and more accurate development.

## Repository Contents:
1. Data Preparation: 
    Scripts and notebooks for preparing and annotating datasets.
2. Model Training:  
    Code for training the detection and classification models.
3. Evaluation: 
    Tools for assessing model performance using metrics like precision, recall, and confusion matrix.


***Here are some of the best object detection transfer learning models widely used in practice due to their accuracy, efficiency, and flexibility:***

1. Faster R-CNN
Overview: A two-stage object detection model where the first stage generates region proposals, and the second stage classifies the proposals and refines bounding boxes.
Best For: High accuracy in detecting objects in complex and cluttered scenes.
Pre-trained Weights: Available on COCO or Pascal VOC datasets.
Transfer Learning Use Case: Fine-tune the network for detecting specific objects in smaller datasets.

2. YOLO (You Only Look Once)
Versions: YOLOv3, YOLOv4, YOLOv5, and YOLOv8 (latest with advancements in speed and accuracy).
Overview: A single-stage detector known for real-time detection speeds and decent accuracy.
Best For: Applications requiring real-time detection with good precision (e.g., surveillance, robotics).
Pre-trained Weights: Provided for COCO and other datasets.
Transfer Learning Use Case: Quick customization for specific object detection tasks.

3. RetinaNet
Overview: A single-stage detector that uses Focal Loss to address class imbalance in object detection.
Best For: Scenarios with small objects or datasets where some classes dominate others.
Pre-trained Weights: Available for COCO dataset.
Transfer Learning Use Case: Fine-tune for datasets with varied object sizes.

4. EfficientDet
Overview: A family of models that balance speed and accuracy using compound scaling.
Best For: Optimizing performance on edge devices or when computational efficiency is critical.
Pre-trained Weights: Available for COCO and Open Images datasets.
Transfer Learning Use Case: Great for mobile or embedded systems with custom datasets.

5. DETR (DEtection TRansformer)
Overview: A transformer-based model that simplifies object detection by removing the need for anchor boxes and post-processing.
Best For: Cutting-edge applications requiring state-of-the-art performance in dense scenes.
Pre-trained Weights: Available on COCO.
Transfer Learning Use Case: Customize the model for complex detection scenarios with fewer annotations.

6. SSD (Single Shot MultiBox Detector)
Overview: A single-stage detector that strikes a balance between speed and accuracy.
Best For: Medium-sized objects and applications where real-time performance is necessary.
Pre-trained Weights: Available for COCO.
Transfer Learning Use Case: Fine-tune for lightweight detection tasks.

7. Mask R-CNN
Overview: An extension of Faster R-CNN that performs object detection and instance segmentation.
Best For: Applications requiring both object detection and pixel-level segmentation.
Pre-trained Weights: Available on COCO.
Transfer Learning Use Case: Fine-tune for detecting and segmenting domain-specific objects.

- **This repository only includes fast R-CNN, yolo, DETR (Detection transformer) and ssd**

**Summary of Selection Based on Needs:**

1. Real-Time Speed: YOLOv5, YOLOv8, SSD.
2. High Accuracy: Faster R-CNN, RetinaNet.
3. Dense or Complex Scenes: DETR, Mask R-CNN.
4. Mobile/Efficient Deployment: EfficientDet, SSD.


# Dataset :

**This dataset has 500 images with 2 classes and annotations labels data.**
    Categories: Two classes – Children's and Adults.
    Diversity: Includes a wide range of scenarios for robust model training.
    Annotations: Provides bounding boxes and class labels for all detected individuals.

![Kaggle link](https://www.kaggle.com/datasets/kaushigihanml/kids-and-adults-detection)
![Roboflow link](https://universe.roboflow.com/children-and-adults-detection/children-and-adults-detection/dataset/2)


# 1. Faster R-CNN - [Notebook]( )
![image]()

This notebook demonstrates the implementation of transfer learning on a Faster R-CNN model using Facebook's Detectron2 library. It includes steps to fine-tune a pre-trained model for custom object detection tasks, with detailed examples of data preparation, training, and evaluation

![Detectron2 ](https://ai.meta.com/tools/detectron2/)
![Detectron2 github ](https://github.com/facebookresearch/detectron2)



# 2. Yolo - [Notebook]( )
![image]()

This Notebook provides an implementation of transfer learning for object detection using the YOLO (You Only Look Once) model. It covers the process of fine-tuning a pre-trained YOLO model on custom datasets, including data preparation, annotation, model configuration, training, and evaluation. Suitable for efficient and real-time object detection tasks.

![yolo](https://docs.ultralytics.com/)
![yolo 8 finetune ](https://docs.ultralytics.com/modes/train/)


# 3. DETR (DEtection TRansformer) - [Notebook]( )
![image]()

This notrbook demonstrates how to implement transfer learning using DETR (DEtection TRansformers) for object detection tasks. It includes steps for fine-tuning a pre-trained DETR model on custom datasets, covering data preparation, model configuration, training, and evaluation. DETR combines transformers with CNNs for end-to-end object detection, enabling improved performance on complex detection tasks.


![Huggingface Detr](https://huggingface.co/docs/transformers/en/model_doc/detr)
![Finetune detr](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR)