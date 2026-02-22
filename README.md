# 🧠 Applied Deep Learning — CV, NLP & Time Series

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-red?logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" alt="Jupyter">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License MIT">
</p>

<p align="center">
  <em>End-to-end deep learning implementations across Computer Vision, Time Series Forecasting, and NLP using TensorFlow/Keras</em>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Results Summary](#-results-summary)
- [Projects Detail](#-projects-detail)
  - [Time Series Forecasting](#1-time-series-forecasting)
  - [CV & NLP Models Overview](#2-cv--nlp-models-overview)
  - [Fashion MNIST Classification](#3-fashion-mnist-classification)
  - [ResNet Implementation](#4-resnet-implementation)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Key Learnings](#-key-learnings)
- [Connect](#-connect)

---

## 📌 Project Overview

This repository showcases hands-on deep learning projects spanning three major domains:

- **Computer Vision:** Image classification from scratch and with state-of-the-art transfer learning (VGG16) on CIFAR-10, plus a survey of modern CV/NLP architectures.
- **Time Series Forecasting:** Comparison of SimpleRNN, LSTM, and GRU on real-world temperature data using a sliding-window approach.
- **Neural Network Fundamentals:** Multi-class clothing classification on Fashion MNIST.

All experiments are implemented end-to-end in Python using TensorFlow/Keras, with clear preprocessing, training, evaluation, and visualization steps.

---

## 📊 Results Summary

| Project | Dataset | Model | Metric | Score |
|---|---|---|---|---|
| Time Series Forecasting | Melbourne Min Temperatures | LSTM / GRU | Val MSE Loss | LSTM & GRU > SimpleRNN |
| Image Classification | CIFAR-10 | VGG16 Fine-tuned | Val Accuracy | **82.31%** |
| Image Classification | Fashion MNIST | Basic Neural Network | Accuracy | **84.5%** |
| Transfer Learning | CIFAR-10 | VGG16 Frozen | Val Accuracy | 50.06% |

---

## 🗂 Projects Detail

### 1. Time Series Forecasting

📓 `Time_Series_Prediction_Comparison.ipynb`

**Description:** Predicts the next day's minimum temperature in Melbourne using a sliding window of 30 previous days.

**Dataset:** Daily Minimum Temperatures in Melbourne (Jason Brownlee's GitHub)

**Approach & Architecture:**
- Sliding window preprocessing (window size = 30)
- MinMaxScaler normalization
- Three models trained and compared: **SimpleRNN**, **LSTM**, **GRU**
  - All models: 50 units, Adam optimizer, MSE loss, 20 epochs, batch size 32
- 90% / 10% train-validation split

**Key Results:**
- LSTM and GRU outperform SimpleRNN on validation MSE loss
- Loss curves and Predictions vs Actual plots generated for all three models

**Technologies:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-learn

---

### 2. CV & NLP Models Overview

📓 `CV_NLP_Models_Overview.ipynb`

**Description:** Demonstrates three different image classification strategies on CIFAR-10, and provides a structured overview of state-of-the-art CV, NLP, and Audio models.

**Dataset:** CIFAR-10 (60,000 images, 10 classes, resized to 224×224)

**Approach & Architecture:**

| Approach | Details | Val Accuracy |
|---|---|---|
| CNN from Scratch | 3 Conv layers + Dense, 3 epochs | ~60.49% |
| VGG16 Frozen (Transfer Learning) | ImageNet weights, all layers frozen, 3 epochs | ~50.06% |
| VGG16 Fine-tuned | Unfreeze last 30% of layers, LR=1e-5, 10 epochs | **82.31%** |

**Architecture Survey Covered:**
- **CV Models:** ResNet, VGG, EfficientNet, MobileNet, DenseNet, Vision Transformer (ViT)
- **NLP Models:** BERT, GPT, RoBERTa, T5, XLM-R
- **Audio Models:** Wav2Vec, Whisper

**Key Results:**
- Fine-tuning VGG16 yields a **+32%** accuracy boost over the frozen baseline
- Highlights the power of transfer learning with domain-specific fine-tuning

**Technologies:** TensorFlow/Keras, TensorFlow Datasets, NumPy, Matplotlib

---

### 3. Fashion MNIST Classification

📓 `Fashion_MNIST_Basic_NN.ipynb`

**Description:** Classifies 10 categories of clothing images using a basic neural network.

**Dataset:** Fashion MNIST (70,000 grayscale 28×28 images, 10 classes)

**Approach & Architecture:**
- Fully connected neural network
- Categorical cross-entropy loss, Adam optimizer

**Key Results:**
- Achieved **~84.5% accuracy** on the test set

**Technologies:** TensorFlow/Keras, NumPy, Matplotlib

---

### 4. ResNet Implementation

📓 `ResNet_Implementation.ipynb`

**Description:** Implements the ResNet (Residual Network) architecture from scratch.

**Approach & Architecture:**
- Residual blocks with skip connections
- Batch normalization and ReLU activations
- Demonstrates how deep networks avoid vanishing gradient issues

**Technologies:** TensorFlow/Keras, NumPy

---

## 🛠 Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-red?logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" alt="Jupyter">
  <img src="https://img.shields.io/badge/NumPy-blue?logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-purple?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-blue" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn" alt="Scikit-learn">
</p>

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmed3li99/Applied-Deep-Learning-TensorFlow.git
cd Applied-Deep-Learning-TensorFlow
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open any `.ipynb` file and run all cells sequentially.

---

## 🧠 Key Learnings

- **Transfer Learning is powerful:** Fine-tuning a pre-trained VGG16 boosted accuracy from ~50% to **82.31%** on CIFAR-10 — a clear demonstration of leveraging pre-learned image features.
- **Recurrent architectures matter:** LSTM and GRU outperform SimpleRNN on time series tasks thanks to their gating mechanisms that better capture long-range dependencies.
- **Data preprocessing is critical:** MinMaxScaler normalization and proper sliding-window construction directly impact model convergence and performance.
- **Depth vs. residual connections:** ResNet's skip connections effectively mitigate vanishing gradients, enabling training of much deeper networks.
- **Architecture selection:** Choosing the right model family (CNN vs. RNN vs. Transformer) is as important as hyperparameter tuning.

---

## 🤝 Connect

I'm available for freelance deep learning and machine learning projects.

- 💼 [Upwork Profile](#) *(link coming soon)*
- 🔗 [LinkedIn](#) *(link coming soon)*
- 📧 Feel free to open an issue or reach out via GitHub

---

<p align="center">
  <em>⭐ If you find this project useful, please consider giving it a star!</em>
</p>

