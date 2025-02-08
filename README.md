# **Letter Recognition**  

## 📌 Project Overview  
This project focuses on recognizing black-and-white pixelated uppercase English letters using machine learning models. The dataset consists of **17,338 samples** generated from 20 different fonts, with random distort
-world variability. The project was developed as part of **COE 292: Introduction to Artificial Intelligence course**  

## 🎯 Objectives  
- Implement and compare different machine learning models for letter recognition.  
- Utilize **feature engineering, dataset preprocessing, and scaling** techniques.  
- Evaluate model performance using **cross-validation and accuracy metrics.**  

## 📂 Dataset  
- **Size:** 17,338 labeled samples  
- **Features:** 16 numerical attributes (statistical moments, edge counts, geometric features)  
- **Labels:** 26 uppercase English letters (A-Z)  
- **Preprocessing:** Feature scaling, outlier removal, and encoding  

## 🛠️ Techniques & Algorithms  
### **1️⃣ K-Nearest Neighbors (KNN)**  
- Used **StandardScaler** for feature normalization.  
- Applied **Principal Component Analysis (PCA)** to reduce dimensionality.  
- **Optimal K = 4** was selected based on cross-validation.  
- **Accuracy:** **90.08%**  

### **2️⃣ Support Vector Machine (SVM)**  
- Used **StandardScaler** for feature scaling.  
- Implemented **Radial Basis Function (RBF) kernel** for non-linear separability.  
- **Accuracy:** **96.46%**  

### **3️⃣ Deep Neural Network (DNN) & Convolutional Neural Network (CNN)**  
- **DNN Architecture:** 3 hidden layers (256, 128, 64 neurons).  
- **Activation Functions:** ReLU (hidden layers), Softmax (output layer).  
- **CNN for improved spatial feature extraction.**  
- **Accuracy:** **97%**  

## 📊 Model Comparison  
| Model | Accuracy | Precision | Recall |  
|--------|---------|------------|---------|  
| KNN | 90.08% | 90% | 90% |  
| SVM | 96.46% | 97% | 96% |  
| CNN | **97%** | **97%** | **97%** |  

## 🚀 Conclusion  
This project demonstrates the effectiveness of **machine learning and deep learning** in recognizing letters from images. While **KNN** is simple but computationally expensive for large datasets, **SVM** provides robust decision boundaries. However, **CNN** achieves the best performance due to its ability to extract spatial features efficiently.  

