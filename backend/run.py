# D:\salary-prediction-ai\backend\run.py
import os
import sys
# Add the parent directory of 'app' to the Python path
# This allows Flask to find the 'app' package correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app import create_app
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler # Import StandardScaler for CustomStandardScaler

# Custom Standard Scaler class - MUST be included here for joblib.load to work
# This needs to be present in the __main__ module (run.py) when the model is loaded
class CustomStandardScaler(StandardScaler):
    def transform(self, X):
        if isinstance(X, pd.Series): # Make sure pd is imported if needed, though it's usually inside prediction_service
            return super().transform(X.values.reshape(-1, 1)).flatten()
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            return super().transform(X.values).flatten()
        elif isinstance(X, np.ndarray) and X.ndim == 1: # Make sure np is imported
            return super().transform(X.reshape(-1, 1)).flatten()
        elif isinstance(X, (np.integer, np.floating)):
            return super().transform(np.array([[X]])).flatten()
        return super().transform(X)

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.Series):
            return super().fit_transform(X.values.reshape(-1, 1)).flatten()
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            return super().fit_transform(X.values).flatten()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            return super().fit_transform(X.reshape(-1, 1)).flatten()
        return super().fit_transform(X, y)

# Ensure pandas and numpy are imported here if CustomStandardScaler directly uses them
# although they are typically imported in prediction_service.py
import pandas as pd
import numpy as np

app = create_app()

# Configure CORS for your Flask app.
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}) # Adjust origin for React dev server

if __name__ == '__main__':
    # You can choose the config class based on environment variables
    # For now, we'll stick to the default in create_app which loads from app.config.Config
    app.run(debug=True, port=5000) # Run on port 5000, enable debug mode for development