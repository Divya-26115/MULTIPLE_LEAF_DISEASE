# GreenVision – Leaf Disease Detection Module

## 📌 Project Overview
GreenVision is an IoT-based Smart Crop Detection & Monitoring system developed as part of an IEEE conference paper (currently under review).

This repository contains the **Leaf Disease Detection module**, which uses a Convolutional Neural Network (CNN) model to classify plant leaf diseases from uploaded images. The system is integrated with a Flask-based web interface 

## 🚀 Features
- Leaf image upload via web interface
- CNN-based disease classification
- Flask web application
- SQLite database for user data storage
- Hardware device connection using serial communication (for system integration)

## 🛠 Technologies Used
- Python
- Flask
- TensorFlow / Keras (CNN Model)
- NumPy
- HTML / CSS
- SQLite
- Machine Learning

## 🧠 System Architecture
The module works as follows:
1. User uploads a plant leaf image.
2. Image is preprocessed and passed to the trained CNN model.
3. Model predicts the disease category.
4. Result is displayed through the Flask web interface.
5. Data can be integrated with IoT monitoring hardware.

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
python app.py
