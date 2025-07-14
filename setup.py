from setuptools import setup, find_packages
import os

setup(
    name='house_price_prediction',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'xgboost==2.0.3',
        'joblib==1.3.2',
        'flask==2.3.2',
    ],
    author='Abhishek Sinha',
    author_email='abhisheksinha.7742@gmail.com',
    description='House Price Prediction Application',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
)