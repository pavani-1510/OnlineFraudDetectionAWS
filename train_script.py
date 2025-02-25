# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define the model_fn function for SageMaker
def model_fn(model_dir):
    """
    This function loads the trained model during deployment.
    Arguments:
    model_dir -- The directory where the model is stored.
    
    Returns:
    model -- The trained RandomForest model.
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

# Define the main function for training the model
if __name__ == '__main__':
    
    # Load the training data from the input directory
    input_data_path = os.path.join('/opt/ml/input/data/train', 'train.csv')
    df = pd.read_csv(input_data_path)
    
    # Split features and target variable
    X_train = df.drop(['target'], axis=1)
    y_train = df['target']
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model in the output directory
    model_dir = '/opt/ml/model'
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
