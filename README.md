🌿 AI Crop Disease Detector

An advanced deep-learning–powered system that detects plant leaf diseases with high accuracy using computer vision and TensorFlow.

🧠 Overview

This project uses a custom-trained MobileNetV2 model fine-tuned on the PlantVillage dataset to automatically identify crop diseases from leaf images.
It provides instant predictions, disease confidence levels, and detailed prevention/treatment guidance through an interactive web interface built with Streamlit.

Farmers, agronomists, and agricultural researchers can use it to monitor crop health, reduce losses, and take early preventive actions against infections.

🚀 Features

🌱 Detects 15+ common crop diseases (Tomato, Potato, Pepper, Apple, Corn, Grape, etc.)

🧩 Trained using heavy real-time data augmentation and class-balanced weighting

📈 Achieved ~97% validation accuracy on test data

🧠 Includes top-3 prediction probabilities for better transparency

💊 Provides disease description, symptoms, prevention, and treatment tips

💻 Integrated Streamlit web UI for easy drag-and-drop leaf analysis

🧾 Compatible with TensorFlow 2.x / Keras

🧪 Model Highlights
Phase	Technique	Accuracy	Notes
Phase 1	Transfer Learning (MobileNetV2)	~41%	Baseline
Phase 2	Fine-Tuning (Unfrozen Top Layers)	~46%	Improved Stability
Phase 3	Heavy Augmentation + Class Weights	~97%	Final Trained Model
📸 Web App Demo

The interface allows users to upload a leaf image and instantly view:

Predicted disease and confidence percentage

Short disease description

Symptom checklist

Prevention & treatment steps

🧰 Tech Stack

Python 3.10+

TensorFlow / Keras

NumPy, Pillow

Stre (for UI)

Matplotlib / Pandas (for analysis)

⚙️ How It Works

Upload a crop leaf image (JPG/PNG).

The model preprocesses and classifies it among trained disease classes.

The app displays the top predictions along with actionable insights.

🌾 Real-World Impact

This tool empowers farmers and agricultural researchers to:

Detect diseases before they spread.

Reduce pesticide misuse through targeted treatment.

Contribute to sustainable agriculture and food security.

🧠 Future Improvements

🌍 Add multilingual support for rural deployment.

📱 Develop a mobile app version using TensorFlow Lite.

🧩 Incorporate satellite & climate data for environmental correlation.

🧬 Expand to more crops beyond the PlantVillage dataset.
