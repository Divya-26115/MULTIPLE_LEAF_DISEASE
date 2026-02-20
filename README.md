# 🌱 GreenVision: Smart Crop Detection & Monitoring System

An IoT-based smart agriculture system designed for real-time crop monitoring, disease detection, and automated irrigation control.


## 📌 Project Overview

GreenVision integrates IoT sensors and Machine Learning to monitor crop health and environmental conditions. The system collects real-time data using ESP32 and environmental sensors, detects leaf diseases using a CNN model, and automates irrigation through a relay mechanism.


## 🚀 Features

- Real-time temperature & humidity monitoring (DHT11)
- Soil moisture detection
- CNN-based crop disease detection
- Automated irrigation using relay module
- Web-based monitoring dashboard (Flask)
- Serial communication between hardware and software


## 🛠️ Technologies Used

Python, Machine Learning (CNN), TensorFlow/Keras, Flask, ESP32/Arduino, IoT Sensors (DHT11, Soil Moisture Sensor), Serial Communication (PySerial), HTML/CSS, SQLite.


## 🏗️ System Architecture

1. Sensors collect environmental data.  
2. ESP32 sends data via serial communication.  
3. Flask web application processes and displays data.  
4. CNN model detects leaf diseases.  
5. Relay module controls irrigation automatically.  


## 👥 Project Team

This project was developed as a group project by:

- Divya M Nagavand
- Keerthi M
- Hannah Susan Blesson
- Gagana S
  

## 📊 Research Work

Research paper titled **"GreenVision: Smart Crop Detection & Monitoring"** submitted to an IEEE International Conference (Under Review).


## 📂 How to Run the Project

1. Clone the repository:
   git clone <your-repo-link>

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Flask application:
   python app.py

4. Connect ESP32 and ensure correct serial port configuration.


## 📌 Future Enhancements

- Cloud integration (AWS / Firebase)  
- Mobile application support  
- Advanced disease classification  
- Real-time farmer alert system  



## 📄 License

This project is developed for academic purposes.
