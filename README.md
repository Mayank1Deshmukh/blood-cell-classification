
---

```markdown
# 🩸 Blood Cell Classification & GAN Image Generation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A professional dark-themed web application to classify blood cell images using ViT, Custom ViT, and Performer models, with GAN-based image generation. Fully tracked using MLflow.

---

## 🌟 Features

- **Image Classification**
  - Predict blood cell types using:
    - Pretrained ViT
    - Custom ViT
    - Performer (Efficient Transformer)
- **GAN Image Generation**
  - Generate synthetic blood cell images for data augmentation or visualization
- **MLflow Integration**
  - Track experiments, metrics, and save models
- **Professional Frontend**
  - Dark-themed UI
  - Real-time predictions and GAN generation

---

## 🖼️ Demo Screenshots

**Upload & Predict:**
![Upload Prediction](screenshots/upload_prediction.png)

**GAN Image Generation:**
![GAN Generation](screenshots/gan_generation.png)

> *Add your screenshots to `screenshots/` folder.*

---

## 📂 Folder Structure

```

blood-cell-classification/
│
├─ app.py                  # Flask API for predictions and GAN
├─ templates/
│   └─ index.html          # Frontend
├─ static/
│   └─ gan_image.png       # Generated GAN image
├─ models/
│   ├─ vit_model.pth
│   ├─ custom_vit_model.pth
│   └─ performer_model.pth
├─ your_model_file.py      # All model classes
├─ notebooks/
│   └─ training_notebooks.ipynb
├─ mlruns/                 # MLflow experiments
├─ requirements.txt
└─ .gitignore

````

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/ankitsunil530/blood-cell-classification.git
cd blood-cell-classification
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

**Features:**

* Upload blood cell images → Predict classes with three models
* Generate GAN images → Real-time synthetic image generation

---

## 🧠 Models

| Model            | Description                                       |
| ---------------- | ------------------------------------------------- |
| **BloodCellViT** | Pretrained ViT-Base/16 fine-tuned on dataset      |
| **CustomViT**    | ViT trained from scratch for 4 blood cell classes |
| **Performer**    | Efficient transformer with linear attention       |
| **GAN**          | DCGAN-based generator for synthetic blood cells   |

---

## 📊 Training & MLflow

* Training notebooks in `notebooks/`
* MLflow logs in `mlruns/`
* Log model & metrics:

```python
mlflow.pytorch.log_model(model, artifact_path="model_name")
mlflow.log_param("learning_rate", 0.0001)
mlflow.log_metric("val_accuracy", 97.12)
```

---

## 🎨 Frontend

* Dark theme with professional look
* Displays:

  * Uploaded image predictions
  * GAN-generated images
* Real-time updates

---

## 📌 Notes

* Place trained model files (`.pth`) in `models/` folder
* MLflow tracking URI:

```python
mlflow.set_tracking_uri("file:///D:/Blood Cell Classifiaction/blood-cell-classification/mlruns")
```

* GAN images saved in `static/gan_image.png`

---

## 🔮 Future Improvements

* Batch predictions for multiple images
* Higher-resolution GAN generation
* Docker deployment for cloud hosting
* User authentication and multi-user support

---

## 📧 Contact

**Sunil Kumar**

* GitHub: [ankitsunil530](https://github.com/ankitsunil530)
* Email: [sunilkumar@example.com](ankitsunil530@.com)

---

## 📝 License

This project is licensed under the MIT License.

```

---
```
