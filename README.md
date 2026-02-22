# 🚀 Deep Learning Applications: CV, NLP & Time Series

This repository demonstrates practical, portfolio-ready implementations of deep learning models across three main domains:
- **Computer Vision (CV)** — image classification with CNNs and transfer learning
- **Time Series Forecasting** — sequence prediction with RNN, LSTM, and GRU
- **Neural Network Fundamentals** — regularization, callbacks, and residual architectures

All notebooks follow a consistent professional structure: EDA → preprocessing → model building → evaluation with metrics → visualizations → Key Takeaways.

---

## 📌 Project Overview

The main objective of this project is to understand and apply deep learning techniques in real-world scenarios by:

- Implementing and comparing recurrent architectures (RNN, LSTM, GRU) for time series prediction using MAE and RMSE
- Building CNNs from scratch and applying transfer learning with VGG16 for CIFAR-10 image classification
- Implementing a mini ResNet with custom residual blocks to address the vanishing gradient problem
- Building a regularized feedforward network (BatchNormalization + Dropout) with smart training callbacks

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📂 Project Structure

| Notebook | Description |
|----------|-------------|
| `Time_Series_Prediction_Comparison.ipynb` | EDA on Melbourne temperature dataset, then trains and evaluates RNN / LSTM / GRU with MAE & RMSE metrics and a final comparison table |
| `CV_NLP_Models_Overview.ipynb` | CIFAR-10 image classification comparing CNN from Scratch, VGG16 Frozen, and VGG16 Fine-tuned; includes seaborn confusion matrix and model comparison table |
| `Fashion_MNIST_Basic_NN.ipynb` | Improved feedforward network with BatchNormalization + Dropout, EarlyStopping + ReduceLROnPlateau callbacks, classification report, and confusion matrix heatmap |
| `ResNet_Implementation.ipynb` | Mini ResNet built from scratch for CIFAR-10 using a custom `residual_block()` function; explains residual connections and the vanishing gradient problem |

---

## 📊 Results

| Notebook | Key Result |
|----------|-----------|
| **Time Series** | LSTM & GRU outperform SimpleRNN; GRU achieves competitive MAE with fewer parameters |
| **CV — CNN Scratch** | ~60% validation accuracy after 3 epochs on CIFAR-10 |
| **CV — VGG16 Frozen** | ~50–65% validation accuracy (limited by domain gap) |
| **CV — VGG16 Fine-tuned** | **~82%** validation accuracy after fine-tuning top 30% of layers |
| **Fashion MNIST** | **88%+** test accuracy using BatchNorm + Dropout + callbacks |
| **Mini ResNet** | ~75–80% test accuracy on CIFAR-10 with only 3 residual blocks |

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmed3li99/Applied-Deep-Learning-TensorFlow.git
cd Applied-Deep-Learning-TensorFlow
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook
```

Then open any `.ipynb` file and run all cells.

---

## 🧠 Key Learnings

- **RNN vs LSTM vs GRU** — gating mechanisms in LSTM/GRU significantly improve long-range dependency capture compared to SimpleRNN
- **Transfer Learning** — fine-tuning a pretrained VGG16 backbone yields ~82% accuracy vs ~60% training from scratch on CIFAR-10
- **Residual Connections** — skip connections in ResNet allow gradients to flow directly, solving the vanishing gradient problem for deep networks (`H(x) = F(x) + x`)
- **Regularization** — BatchNormalization + Dropout together push Fashion MNIST accuracy above 88%
- **Training Callbacks** — EarlyStopping and ReduceLROnPlateau prevent overfitting and enable efficient training without manual tuning
- **Evaluation** — using MAE, RMSE, classification reports, and confusion matrices provides a complete picture of model performance

