**🌿 Crop Disease Detection using Deep Learning & Flask**

An AI-powered web application that detects plant leaf diseases using a TensorFlow deep learning model (MobileNetV2) and provides prevention & treatment suggestions.
Built with Flask for the backend and HTML/CSS for a clean and interactive user interface.

**🚀 Features**

🧠 Deep Learning Model (97% Accuracy) — identifies 15+ crop leaf diseases.

🌱 Flask Web App — lightweight, fast, and easy to deploy locally or on cloud.

🎨 HTML/CSS Frontend — intuitive design for smooth user experience.

📸 Image Upload Interface — upload a leaf photo to detect the disease instantly.

💊 Smart Insights — prevention and treatment measures for each detected disease.

**🧩 Tech Stack**

Frontend: HTML, CSS

Backend: Flask (Python)

AI/Model: TensorFlow, Keras (MobileNetV2)

Tools: NumPy, Pillow, ImageDataGenerator

**⚙️ How It Works**

User uploads a leaf image through the web interface.

Flask backend processes and feeds the image into the trained deep learning model.

The model predicts the disease with confidence levels.

The result page displays the top prediction with detailed prevention and treatment info.

**📸 Demo Output**
🏆 Tomato Late Blight  
Confidence: 96.2%  
Prevention: Use resistant varieties, apply fungicides, remove infected plants.  
Treatment: Copper fungicides or Mancozeb spray.  

**🧠 Model Training Highlights**

Base model: MobileNetV2

Data Augmentation: Heavy rotation, flips, zoom, brightness

Optimization: Adam optimizer + Label smoothing

Result: Achieved 97.6% validation accuracy

**🧰 Setup Instructions**
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/crop-disease-detection.git

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run Flask app
python app.py

# 4️⃣ Open in browser
http://127.0.0.1:5000

**📬 Future Enhancements**

🌾 Add more crop categories

📱 Build responsive mobile UI

☁️ Deploy on AWS / Render / Heroku

**🧑‍💻 Author**

Binil John
Deep Learning | Computer Vision | Flask Web Developer
