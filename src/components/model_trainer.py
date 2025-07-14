import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import os
from src.logger import logging
from src.exception import CustomException
import sys

class ModelTrainer:
    def __init__(self):
        self.artifacts_path = os.path.join('artifacts')
        self.model_path = os.path.join(self.artifacts_path, 'model.pkl')
        
    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training")
            
            # Define models
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                ),
                'XGBoost': XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.01,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            }
            
            # Train and evaluate models
            best_model = None
            best_r2 = -float('inf')
            best_model_name = None
            
            for name, model in models.items():
                logging.info(f"Training {name} model")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_exp = np.expm1(y_test)
                y_pred_exp = np.expm1(y_pred)
                r2 = r2_score(y_test_exp, y_pred_exp)
                
                logging.info(f"{name} R² Score: {r2:.2f}")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = name
            
            # Save best model
            joblib.dump(best_model, self.model_path)
            logging.info(f"Best model ({best_model_name}) saved at {self.model_path}")
            
            return best_model, best_model_name
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)