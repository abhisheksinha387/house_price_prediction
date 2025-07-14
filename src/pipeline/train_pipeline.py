from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import sys
import os

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            
            # Data ingestion
            df = self.data_ingestion.initiate_data_ingestion()
            
            # Data transformation
            X_transformed, y, preprocessor = self.data_transformation.initiate_data_transformation(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y, test_size=0.2, random_state=42
            )
            
            # Model training
            best_model, best_model_name = self.model_trainer.initiate_model_training(
                X_train, y_train, X_test, y_test
            )
            
            logging.info(f"Training pipeline completed. Best model: {best_model_name}")
            
            return best_model, best_model_name
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()