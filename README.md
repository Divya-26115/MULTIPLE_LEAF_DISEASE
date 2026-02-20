# GreenVision – Leaf Disease Detection & IoT Integration Module

## 📌 Project Overview
GreenVision is an IoT-based Smart Crop Detection & Monitoring system developed as part of an IEEE conference paper (currently under review).

This repository contains the Leaf Disease Detection module integrated with hardware-based environmental monitoring using serial communication.

The system combines CNN-based image classification with real-time sensor data collection from IoT hardware.

## 🚀 Features
- Leaf image upload via Flask web interface
- CNN-based plant disease classification
- Real-time prediction results
- Serial communication with hardware module
- Integration with environmental sensors (soil moisture, temperature, humidity)
- SQLite database for user data storage

## 🔌 Hardware Integration
The system communicates with IoT hardware (ESP32/Arduino) using serial communication.

Sensors used:
- DHT11 (Temperature & Humidity)
- Soil Moisture Sensor
- Relay module for irrigation control

Sensor data is transmitted to the web application for monitoring and decision-making.

## 🛠 Technologies Used
- Python
- Flask
- TensorFlow / Keras (CNN)
- NumPy
- SQLite
- HTML / CSS
- Serial Communication (PySerial)
- IoT Hardware (ESP32 / Arduino)

## 🧠 Working Principle
1. Sensors collect environmental data.
2. Data is transmitted via serial communication to the Flask application.
3. User uploads leaf image for disease prediction.
4. CNN model classifies the disease.
5. System can assist in smart irrigation decisions.

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
