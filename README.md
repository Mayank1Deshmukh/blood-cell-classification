# 🩸 Blood Cell Type Classification

## 📘 Project Overview
This project focuses on **automatically classifying blood cell images** into different types such as **Eosinophil, Lymphocyte, Monocyte, and Neutrophil** using deep learning models.

We implemented and compared:
- **Vision Transformer (ViT)**
- **Performer Transformer**

The goal is to evaluate both models on metrics like **accuracy, training time, inference time, and trainable parameters**, and then deploy the best-performing model using **Streamlit**.

---

## 🎯 Objectives
- Classify RBC, WBC, and Platelets (using WBC subtypes from the dataset)
- Compare ViT and Performer models
- Track and manage experiments with **MLflow**
- Version control datasets and models using **DVC**
- Deploy the best model as a web app (Streamlit)

---

## 🧠 Dataset
**Dataset:** [Blood Cells Dataset - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

**Contains:**
- 4 blood cell types:
  - `EOSINOPHIL`
  - `LYMPHOCYTE`
  - `MONOCYTE`
  - `NEUTROPHIL`

