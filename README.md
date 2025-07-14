# 🏠 House Price Prediction

📍 **Live Demo:** [Click to View on Hugging Face Spaces 🚀](https://huggingface.co/spaces/abhisheksinha7742/house_price_prediction)

A machine learning project to predict house prices based on features like area, number of bedrooms, bathrooms, stories, parking, and amenities. It features a full ML pipeline, model training, and a user-friendly Flask web interface—ready to deploy in Docker or Hugging Face Spaces.

---

## 🚀 Features

- 📊 End-to-end ML pipeline (ingestion → transformation → training → prediction)
- 🧠 Feature engineering with smart derived metrics
- 🤖 Model selection: Random Forest, Gradient Boosting, XGBoost
- 🪵 Logging and custom error handling
- 🌐 Flask web interface for real-time predictions
- 📦 Docker container support for deployment
- 🔌 Hugging Face Spaces live demo integration

---

## 📁 Project Structure

```text
house_price_prediction/
├── notebooks/
│   └── Housing.csv            # Raw dataset
├── src/
│   ├── components/            # Core pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/              # Training & prediction pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── exception.py           # Custom exception handling
│   └── logger.py              # Logging configuration
├── templates/
│   └── home.html              # HTML template for Flask UI
├── app.py                     # Flask application
├── Dockerfile                 # Docker setup
├── requirements.txt           # Python dependencies
├── setup.py                   # Packaging setup
└── .gitignore                 # Ignored files

---

## ⚙️ Installation & Setup

### ✅ Prerequisites

* Python 3.9+
* `pip`
* Docker (optional)

### 🔧 Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/house_price_prediction.git
   cd house_price_prediction
   ```

2. **Install the dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**

   ```bash
   python src/pipeline/train_pipeline.py
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```

5. **Open in browser**

   ```
   http://localhost:7860
   ```

---

## 🐳 Docker Deployment

> Run the app in a containerized environment

1. **Build Docker image**

   ```bash
   docker build -t house-price-prediction .
   ```

2. **Start container**

   ```bash
   docker run -p 7860:7860 house-price-prediction
   ```

3. **Access app**

   ```
   http://localhost:7860
   ```

---

## 💡 Usage Instructions

1. Open the app in your browser.
2. Fill in the house features:

   * 📐 Area (in sq. ft.)
   * 🛏️ Bedrooms
   * 🛁 Bathrooms
   * 🏢 Stories
   * 🚗 Parking spots
   * ✅ Amenities (main road, guest room, basement, etc.)
   * 🛋️ Furnishing status
3. Click **“Predict Price”** to view the estimated house value.

---

## 🧠 Models Compared

* 🌲 Random Forest Regressor
* 🔥 Gradient Boosting Regressor
* ⚡ XGBoost Regressor

> The model with the best performance is selected automatically for prediction.

---

## 🧪 Feature Engineering

Derived features include:

* Area per bedroom
* Bedroom-to-bathroom ratio
* Total room count
* Parking per room
* Amenities count
* Luxury score based on amenity presence
* Area per story
* Luxury area (luxury score × area)
* Interaction terms (e.g. stories × parking)

---

## 📋 Logging

* All activities are logged to the `logs/` directory.
* Helpful for debugging, model tracking, and audit trails.

---

## 📄 License

Licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for full details.

---

## 👨‍💻 Author

**Abhishek Sinha**
📧 [abhisheksinha.7742@gmail.com](mailto:abhisheksinha.7742@gmail.com)
🔗 [Hugging Face Profile](https://huggingface.co/spaces/abhisheksinha7742)

---

> 💬 *Pull requests, issues, and feedback are welcome!*

```

---


