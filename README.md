
# Live Demo : https://huggingface.co/spaces/abhisheksinha7742/house_price_prediction

```markdown
# 🏠 House Price Prediction

A machine learning project that predicts house prices based on features such as area, number of bedrooms, bathrooms, stories, amenities, and more. This project includes a complete pipeline from data ingestion to model deployment with a Flask web interface.

---

## 🚀 Features

- ✅ End-to-end ML pipeline (data ingestion, transformation, training)
- ✅ Feature engineering with derived predictors
- ✅ Model selection: Random Forest, Gradient Boosting, XGBoost
- ✅ Logging & custom exception handling
- ✅ Flask web interface for predictions
- ✅ Docker support for containerized deployment

---

## 📁 Project Structure

```

house\_price\_prediction/
├── notebooks/
│   └── Housing.csv            # Raw dataset
├── src/
│   ├── components/            # Core pipeline components
│   │   ├── data\_ingestion.py
│   │   ├── data\_transformation.py
│   │   └── model\_trainer.py
│   ├── pipeline/              # Training and prediction pipelines
│   │   ├── train\_pipeline.py
│   │   └── predict\_pipeline.py
│   ├── exception.py           # Custom exception handling
│   └── logger.py              # Logging configuration
├── templates/
│   └── home.html              # HTML for Flask frontend
├── app.py                     # Main Flask app
├── Dockerfile                 # Docker setup
├── requirements.txt           # Dependencies list
├── setup.py                   # Package configuration
└── .gitignore

````

---

## ⚙️ Installation & Setup

### ✅ Prerequisites

- Python 3.9+
- `pip`
- Docker (optional, for containerization)

### 🔧 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/house_price_prediction.git
   cd house_price_prediction
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run training pipeline:**

   ```bash
   python src/pipeline/train_pipeline.py
   ```

4. **Launch the Flask app:**

   ```bash
   python app.py
   ```

5. **Visit the web interface:**

   ```
   http://localhost:7860
   ```

---

## 🐳 Docker Deployment

1. **Build the Docker image:**

   ```bash
   docker build -t house-price-prediction .
   ```

2. **Run the Docker container:**

   ```bash
   docker run -p 7860:7860 house-price-prediction
   ```

3. **Access the application at:**

   ```
   http://localhost:7860
   ```

---

## 💡 Usage

1. Open the web interface.
2. Enter house details:

   * Area (sq. ft.)
   * Bedrooms
   * Bathrooms
   * Stories
   * Amenities (main road, guest room, basement, etc.)
   * Parking spots
   * Furnishing status
3. Click **"Predict Price"** to view the estimated value.

---

## 🤖 Models Used

The app compares multiple regression models and selects the best one based on performance:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

## 🧪 Feature Engineering

Advanced feature creation includes:

* Area per bedroom
* Bedroom-to-bathroom ratio
* Total room count
* Parking per room
* Luxury score based on amenities
* Area per story
* Amenities count
* Luxury area (luxury score × area)
* Stories × parking interaction

---

## 📝 Logging

All logs are stored in the `logs/` directory with timestamps. This helps with debugging and tracking pipeline runs.

---

## 🪪 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abhishek Sinha**
📧 [abhisheksinha.7742@gmail.com](mailto:abhisheksinha.7742@gmail.com)

```


