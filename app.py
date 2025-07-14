from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys
import os
import logging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = CustomData(
            area=float(request.form['area']),
            bedrooms=int(request.form['bedrooms']),
            bathrooms=int(request.form['bathrooms']),
            stories=int(request.form['stories']),
            mainroad=request.form['mainroad'],
            guestroom=request.form['guestroom'],
            basement=request.form['basement'],
            hotwaterheating=request.form['hotwaterheating'],
            airconditioning=request.form['airconditioning'],
            parking=int(request.form['parking']),
            prefarea=request.form['prefarea'],
            furnishingstatus=request.form['furnishingstatus']
        )
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=f"${prediction[0]:,.2f}")
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('home.html', error="An error occurred while processing your request. Please try again.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)