# 🧠 MindScope AI — Mental Health Classifier using BERT

A deep learning-based NLP classifier that detects signs of mental health distress in Reddit posts. This project leverages the power of **BERT** to analyze textual data and predict whether a post reflects mental health challenges.

---

## 🧾 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset & Credits](#-dataset--credits)
- [Model Performance](#-model-performance)
- [How to Use](#-how-to-use)
- [API Deployment](#-api-deployment)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🧠 Overview

Mental health concerns often go unnoticed on social platforms. This project aims to help in early detection of psychological distress by analyzing written text. Built using **transformers (BERT)** and fine-tuned on a Reddit-based mental health dataset, this model can:

- Classify Reddit posts into `depressed` or `not depressed`
- Be used as a backend for mental health apps
- Provide APIs for real-time inference

---

## ✨ Features

- ✅ Fine-tuned `bert-base-uncased` using PyTorch
- ✅ Custom Dataset & DataLoader handling
- ✅ Tokenization using Hugging Face
- ✅ Trained on real Reddit mental health posts
- ✅ FastAPI backend for deployment
- ✅ Optional Streamlit UI for demo/testing

---

## ⚙️ Tech Stack

| Component         | Used Tool               |
|------------------|--------------------------|
| Language          | Python 3.10              |
| Model             | BERT (Hugging Face)      |
| Training Framework| PyTorch                  |
| Deployment        | FastAPI / Streamlit      |
| Visualization     | Matplotlib / Seaborn     |
| Cloud Training    | Google Colab             |
| Dataset           | Reddit Mental Health     |

---

## 📂 Dataset & Credits

This project uses the **Reddit Mental Health Dataset (RMHD)** sourced from Kaggle:

> 📝 **Dataset**: [Reddit Mental Health Dataset (Kaggle)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)  
> 📌 **License**: For academic and research purposes  
> 📊 **Details**: Includes preprocessed Reddit posts with LIWC, sentiment, and TF-IDF features.  
> 🧑‍💻 **Authors**: Saima R.

Special thanks to the authors and contributors of the dataset.

---

## 📈 Model Performance

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | ~72%      |
| F1 Score     | ~0.84     |
| Model Type   | BERT      |
| Classes      | 2 (Depressed / Not Depressed) |

> 📌 Fine-tuned on ~70% train / 30% test split  
> 📉 Loss and metrics visualizations available in notebook

---

## 🚀 How to Use

### 🔧 Clone the Repository

```bash
git clone https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset
cd Mental_health_AI
