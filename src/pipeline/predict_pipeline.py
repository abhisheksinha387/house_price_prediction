import pandas as pd
import numpy as np
import joblib
import os
from src.logger import logging
from src.exception import CustomException
import sys

class CustomData:
    
    def __init__(self, area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                 basement, hotwaterheating, airconditioning, parking, prefarea, 
                 furnishingstatus):
        self.artifacts_path = '/app/artifacts'  # Explicitly set to /app/artifacts
        self.model_path = os.path.join(self.artifacts_path, 'model.pkl')
        self.preprocessor_path = os.path.join(self.artifacts_path, 'preprocessor.pkl')
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.stories = stories
        self.mainroad = mainroad
        self.guestroom = guestroom
        self.basement = basement
        self.hotwaterheating = hotwaterheating
        self.airconditioning = airconditioning
        self.parking = parking
        self.prefarea = prefarea
        self.furnishingstatus = furnishingstatus
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'area': [self.area],
                'bedrooms': [self.bedrooms],
                'bathrooms': [self.bathrooms],
                'stories': [self.stories],
                'mainroad': [self.mainroad],
                'guestroom': [self.guestroom],
                'basement': [self.basement],
                'hotwaterheating': [self.hotwaterheating],
                'airconditioning': [self.airconditioning],
                'parking': [self.parking],
                'prefarea': [self.prefarea],
                'furnishingstatus': [self.furnishingstatus]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                                'airconditioning', 'prefarea', 'furnishingstatus']
            for col in categorical_columns:
                df[col] = df[col].astype('category')
                
            return df
            
        except Exception as e:
            logging.error(f"Error in creating custom data frame: {str(e)}")
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        self.artifacts_path = os.path.join('artifacts')
        self.model_path = os.path.join(self.artifacts_path, 'model.pkl')
        self.preprocessor_path = os.path.join(self.artifacts_path, 'preprocessor.pkl')
        
    def create_features(self, df):
        try:
            df['area_per_bedroom'] = df['area'] / df['bedrooms']
            df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
            df['total_rooms'] = df['bedrooms'] + df['bathrooms']
            df['parking_per_room'] = df['parking'] / (df['bedrooms'] + df['bathrooms']).replace(0, 1)
            df['luxury_score'] = (df['airconditioning'].cat.codes +
                                df['hotwaterheating'].cat.codes * 2 +
                                df['basement'].cat.codes +
                                df['guestroom'].cat.codes +
                                df['prefarea'].cat.codes * 2)
            df['area_per_story'] = df['area'] / df['stories'].replace(0, 1)
            df['amenities_count'] = df[['mainroad', 'guestroom', 'basement', 
                                      'hotwaterheating', 'airconditioning', 
                                      'prefarea']].eq('yes').sum(axis=1)
            df['luxury_area'] = df['luxury_score'] * df['area']
            df['stories_parking'] = df['stories'] * df['parking']
            return df
        except Exception as e:
            logging.error(f"Error in feature creation: {str(e)}")
            raise CustomException(e, sys)
    
    def predict(self, df):
        try:
            logging.info("Starting prediction")
            
            # Create features
            df = self.create_features(df)
            
            # Load model and preprocessor
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)
            
            # Transform data
            data_transformed = preprocessor.transform(df)
            
            # Make prediction
            prediction = model.predict(data_transformed)
            prediction = np.expm1(prediction)
            
            logging.info("Prediction completed")
            return prediction
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)