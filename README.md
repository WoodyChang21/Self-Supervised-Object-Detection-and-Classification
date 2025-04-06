# Insect Detection and Classification using DINOv2 and EfficientNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dWg2FAhULTnaJLIVpw-RCQ-cxaUDvtF_)

This project demonstrates an **end-to-end insect detection and classification pipeline** using:

- 🧠 **DINOv2 (Self-Supervised Vision Transformer)** for unsupervised object localization via attention maps  
- 📦 **EfficientNet-B0** as a high-accuracy classifier for insect species

The approach leverages DINOv2's ability to detect object regions **without ground-truth labels**, followed by classification using a fine-tuned EfficientNet model. The entire pipeline is implemented in [PyTorch](https://pytorch.org/) and runs fully in [Google Colab](https://colab.research.google.com/).

---



## 🔍 Motivation

Manual annotation of insect bounding boxes is time-consuming and expensive. This project explores **label-free detection** using DINOv2 attention maps, which highlight object regions automatically. Once the objects are cropped, they are passed to a powerful classifier to recognize the insect species.

---

## 📈 Pipeline Overview

1. **Object Detection (DINOv2 Attention Maps):**  
   Uses the [CLS] token’s attention weights to identify and localize regions of interest.

2. **Attention-Guided Cropping:**  
   Applies a threshold and morphological filtering to generate bounding boxes around attention hotspots.

3. **Classification (EfficientNet-B0):**  
   Each cropped insect region is passed into a fine-tuned EfficientNet model for classification.

---

## 🖼️ Sample Output

Here is a visual output of the full pipeline:

<p align="center">   

  ![image](https://github.com/user-attachments/assets/9071dfff-4b85-467d-8b4b-a7c4c0ecee11)<br>
</p>

---


## 🎯 Results

- **Classification**
  - Evaluated quantitatively on a labeled dataset.
  - Best model: EfficientNet-B0
  - Accuracy: **99.2%**

- **Detection (DINOv2)**
  - Evaluated qualitatively due to lack of ground-truth boxes.
  - Visual inspection shows accurate cropping in most cases.
  - Assumption: *Correct cropping → correct classification*

<p align="center">
  <img src="https://i.imgur.com/your_image_link_here.png" width="500"/>
  <br><em>Sample output: Red boxes indicate attention-based object regions, green labels show classified insect types</em>
</p>

---

## 🛠️ Key Techniques

### 🧠 DINOv2 Attention Visualization

We extract attention maps from a specific transformer layer using:

- **CLS attention to all patches**
- **Averaging and max pooling across attention heads**
- **Upsampling to image resolution**
- **Morphological filtering (dilation, erosion)**
- **Bounding box extraction via OpenCV**

### 🧪 EfficientNet-B0 Classification

After cropping detected insect regions, we classify them using a pre-trained EfficientNet-B0 model, which offers:

- Strong performance on small datasets
- High parameter efficiency
- Fast inference time

---

## 📌 Why EfficientNet-B0?

- Lightweight and accurate
- Easy to fine-tune on small datasets
- Outperformed AlexNet and ResNet variants in our experiments
- Larger variants (B1–B7) were unnecessary for this problem scale

---

## 📍 Future Work

- 🔧 Improve object detection via fine-tuning or using DINOv2 visual features (not just attention)
- 🧪 Explore end-to-end training with detection + classification jointly
- 📦 Add evaluation using real bounding box ground-truth (when available)
- 🎯 Test on other fine-grained classification datasets

---

## 📋 Citation and Acknowledgements

- [DINOv2 by Meta AI](https://github.com/facebookresearch/dinov2)
- [EfficientNet PyTorch Implementation](https://github.com/lukemelas/EfficientNet-PyTorch)

Please ⭐ the repo if you found it useful!

---

**Try the notebook in Colab 👉 [Open in Colab](https://colab.research.google.com/drive/1dWg2FAhULTnaJLIVpw-RCQ-cxaUDvtF_)**

**Link to our dataset (https://drive.google.com/drive/folders/1Q4Kbpxe4MIyGlCfuXiIHoXk3fzNCS6Gz?usp=sharing)**
