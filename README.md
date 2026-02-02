# 🌿 Plant Disease Detection System

An end-to-end deep learning system for **plant leaf disease detection** using **transfer learning** and **computer vision**.  
The project includes **model training, fine-tuning, evaluation, and a FastAPI backend** for real-time image-based disease prediction.

---

## 📌 Problem Statement

Plant diseases significantly affect agricultural productivity and food security. Early and accurate detection of plant diseases can help farmers take timely preventive measures.

Traditional disease identification requires expert knowledge and manual inspection, which is:
- Time-consuming  
- Error-prone  
- Not scalable  

This project aims to **automate plant disease detection** using deep learning and deploy the solution as a **real-time prediction API**.

---

## 🎯 Project Objectives

- Build a **multi-class plant disease classifier** using CNNs  
- Leverage **transfer learning** to improve accuracy and efficiency  
- Optimize training using **GPU-accelerated TensorFlow pipelines**  
- Serve predictions via a **FastAPI backend**  
- Provide confidence-aware predictions suitable for real-world use  
## 📊 Dataset

- **Source:** Kaggle – New Plant Diseases Dataset (Augmented)  
- **Number of Classes:** 38  
- **Images:** ~87,000 RGB leaf images  
- **Structure:**






> Note: The test set is intentionally ignored due to insufficient samples. Validation data is used for evaluation.

---

## 🧠 Solution Overview

The system follows an end-to-end machine learning workflow:

1. Dataset ingestion & preprocessing  
2. CNN training using transfer learning  
3. Model evaluation and visualization  
4. Model fine-tuning for better generalization  
5. Deployment via FastAPI for real-time inference  

---

## 🏗️ Project Architecture

plant-disease-detection-system/
│
├── app/ # FastAPI backend
│ ├── init.py
│ ├── main.py # API routes
│ └── model_utils.py # Model loading & inference logic
│
├── model/ # Trained model artifacts
│ ├── plant_disease_model.keras
│ └── class_names.pkl
│
├── notebooks/ # Training & experimentation notebooks
│
├── requirements.txt # Backend dependencies
├── README.md # Project documentation
└── sample_images/ # Images for testing the API


## 🧪 Model Details

### 🔹 Architecture
- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Approach:** Transfer Learning  
- **Classifier Head:**
  - Global Average Pooling  
  - Dense layer (ReLU)  
  - Dropout (regularization)  
  - Softmax output layer (38 classes)  

### 🔹 Training Strategy
- Initial training with frozen backbone  
- Optimized TensorFlow `tf.data` pipeline  
- GPU acceleration (Google Colab T4)  
- Fine-tuning of upper CNN layers with low learning rate  

---

## 📈 Results & Performance

- **Validation Accuracy:** ~95%  
- Stable convergence with minimal overfitting  
- Confusion matrix and classification report used for evaluation  
- Improved generalization across non-dominant classes after fine-tuning  

---

## 🔍 Evaluation & Visualization

The project includes:
- Training vs validation accuracy curves  
- Training vs validation loss curves  
- Confusion matrix for class-wise performance  
- Classification report (Precision, Recall, F1-score)  
- Visual inspection of sample predictions  


## 🚀 FastAPI Backend

The trained model is deployed as a **REST API** using FastAPI.

### 🔹 Features
- Image upload endpoint  
- Real-time disease prediction  
- Confidence score for predictions  
- Swagger UI for easy testing  

---

### 🔹 API Endpoints

#### `GET /`
Health check endpoint

```json
{
  "message": "Plant Disease Detection API is running"
}

{
  "disease": "Tomato___Late_blight",
  "confidence": 0.97
}
```



# 📘 — TECH STACK, LEARNINGS & FUTURE


## ⚙️ Technologies Used

- Python  
- TensorFlow / Keras  
- MobileNetV2  
- NumPy  
- Matplotlib / Seaborn  
- FastAPI  
- Uvicorn  
- Pillow  
- Google Colab (GPU)  

---

## 📌 Key Learnings

- Practical application of transfer learning  
- Handling large-scale image datasets  
- Optimizing GPU training pipelines  
- Debugging model bias and data imbalance  
- Deploying deep learning models as APIs  
- Bridging the gap between ML research and production  

---

## 🔮 Future Enhancements

- Class imbalance handling using class weights  
- Grad-CAM based explainability  
- Top-3 predictions with uncertainty handling  
- Web frontend using Streamlit or React  
- Cloud deployment (Render / Hugging Face Spaces)  
- Mobile deployment using TensorFlow Lite  


---

## 👤 Author

**Utkrisht Naman**  
Aspiring Data Scientist / AI-ML Engineer  

---

## ⭐ Acknowledgements

- Kaggle community for the dataset  
- TensorFlow & FastAPI documentation  
- Open-source contributors  



