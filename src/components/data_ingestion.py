import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self):
        self.data_path = os.path.join('notebooks', 'Housing.csv')
        self.artifacts_path = os.path.join('artifacts')
        os.makedirs(self.artifacts_path, exist_ok=True)
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")
            df = pd.read_csv(self.data_path)
            
            # Convert categorical columns to category type
            categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                                'airconditioning', 'prefarea', 'furnishingstatus']
            for col in categorical_columns:
                df[col] = df[col].astype('category')
                
            # Save raw data
            raw_data_path = os.path.join(self.artifacts_path, 'raw_data.csv')
            df.to_csv(raw_data_path, index=False)
            logging.info(f"Raw data saved at {raw_data_path}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)