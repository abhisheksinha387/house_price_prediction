import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
from src.logger import logging
from src.exception import CustomException
import sys

class DataTransformation:
    def __init__(self):
        self.artifacts_path = os.path.join('artifacts')
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
    
    def get_data_transformer_object(self):
        try:
            numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                              'area_per_bedroom', 'bed_bath_ratio', 'total_rooms',
                              'parking_per_room', 'luxury_score', 'area_per_story',
                              'amenities_count', 'luxury_area', 'stories_parking']
            categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                                 'airconditioning', 'prefarea', 'furnishingstatus']
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            return preprocessor
            
        except Exception as e:
            logging.error(f"Error in creating preprocessor: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, df):
        try:
            logging.info("Starting data transformation")
            
            # Handle outliers
            q_low = df["price"].quantile(0.01)
            q_hi = df["price"].quantile(0.99)
            df = df[(df["price"] < q_hi) & (df["price"] > q_low)]
            
            # Create features
            df = self.create_features(df)
            
            # Separate features and target
            X = df.drop('price', axis=1)
            y = np.log1p(df['price'])
            
            # Get preprocessor
            preprocessor = self.get_data_transformer_object()
            
            # Transform data
            X_transformed = preprocessor.fit_transform(X)
            
            # Save preprocessor
            joblib.dump(preprocessor, self.preprocessor_path)
            logging.info(f"Preprocessor saved at {self.preprocessor_path}")
            
            return X_transformed, y, preprocessor
            
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)