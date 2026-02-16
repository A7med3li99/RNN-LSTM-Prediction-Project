# RNN-LSTM-Prediction-Project
🧠 Deep Learning Applications: Computer Vision, NLP, and Time Series Analysis

## Project Overview

This repository showcases a comprehensive deep learning project covering core concepts in Computer Vision (CV), Natural Language Processing (NLP) models, and practical Time Series Forecasting. The goal was to implement, compare, and analyze various state-of-the-art neural network architectures from scratch using TensorFlow/Keras, demonstrating proficiency in both classification and sequence modeling tasks.

---

## 🎯 Key Features & Technical Highlights

*   **Advanced Model Implementation:** Successfully implemented and analyzed modern CNN architectures, including a focus on **ResNet** and exploration of **VGG16**, for image feature extraction and classification.
*   **Time Series Forecasting:** Conducted a rigorous comparative study on Recurrent Neural Networks (**RNN, LSTM, and GRU**) for a 30-day minimum temperature prediction task, identifying the optimal model for sequential data.
*   **Fundamentals to Advanced:** Built and trained a foundational Dense Neural Network (NN) on the **Fashion MNIST** dataset, demonstrating mastery of key DL concepts like data preprocessing, normalization, and model compilation.
*   **Preprocessing Pipelines:** Implemented advanced data preparation techniques, including image resizing, pixel normalization, and data augmentation (`tf.image.random_flip_left_right`).

---

## 📂 Repository Structure

The project is organized into modular Jupyter Notebooks, each focusing on a specific deep learning area:

| File Name | Primary Focus | Models/Concepts Covered |
| :--- | :--- | :--- |
| `Time_Series_Prediction_Comparison.ipynb` | Time Series Forecasting (Temperature) | **RNN, LSTM, GRU** comparison, Data Scaling. |
| `ResNet_Implementation.ipynb` | Computer Vision - Deep CNN Architecture | **ResNet** structure and implementation. |
| `Fashion_MNIST_Basic_NN.ipynb` | Deep Learning Fundamentals & Classification | Sequential API, Dense Layers, Fashion MNIST, Basic Evaluation. |
| `CV_NLP_Models_Overview.ipynb` | Conceptual Overview of State-of-the-Art Models | **BERT, GPT, T5, EfficientNet, MobileNet** (SOTA models in CV/NLP). |

---

## 🛠 Technologies Used

*   **Python:** Programming Language
*   **TensorFlow & Keras:** Core Deep Learning Frameworks
*   **NumPy & Pandas:** Data Manipulation and Preprocessing
*   **Matplotlib:** Data and Results Visualization (e.g., training history, prediction vs. actual plots)
*   **Scikit-learn:** For performance metrics (e.g., classification report).

---

## 📈 Measurable Results (Key Findings)

*   **Fashion MNIST Classification:** Achieved a test accuracy of **~84.5%** with a validation loss of **~0.4130** using a simple Sequential Dense Network.
*   **Sequence Modeling Performance:** The comparative analysis provided insights into the superior performance and stability of **LSTM/GRU** over a standard RNN for time-dependent sequence prediction.
*   **Code Quality:** All models are built with clean, well-documented code, ensuring reproducibility and clarity on hyperparameter tuning and model configuration (e.g., Adam/SGD optimizers, various activation functions like ReLU, Sigmoid, Tanh).

---

## 🚀 How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```
2.  **Install dependencies:**
    *(Assumes you have Python and pip installed)*
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn jupyter
    ```
3.  **Run the Notebooks:**
    ```bash
    jupyter notebook
    ```
    Open any `.ipynb` file to run the code cells and view the results.
