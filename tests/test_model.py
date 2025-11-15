import pytest
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier

def test_model_file_exists():
    """Test that the trained model file exists"""
    assert os.path.exists('models/knn_model.pkl'), "Model file not found"

def test_model_loads_successfully():
    """Test that the model can be loaded without errors"""
    model = joblib.load('models/knn_model.pkl')
    assert model is not None
    assert isinstance(model, KNeighborsClassifier)

def test_model_predictions_are_valid():
    """Test that model predictions are valid iris classes (0, 1, or 2)"""
    model = joblib.load('models/knn_model.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    predictions = model.predict(X_test)
    
    # check all predictions are valid classes
    assert all(pred in [0, 1, 2] for pred in predictions)
    assert len(predictions) == len(X_test)

def test_model_accuracy_threshold():
    """Test that model accuracy meets minimum threshold"""
    model = joblib.load('models/knn_model.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    
    # minimum acceptable accuracy for iris dataset
    assert accuracy >= 0.85, f"Model accuracy {accuracy} below threshold 0.85"

def test_model_input_output_shapes():
    model = joblib.load('models/knn_model.pkl')
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # test single sample
    single_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=columns)
    prediction = model.predict(single_input)
    assert prediction.shape == (1,)
    
    # test multiple samples
    multiple_input = pd.DataFrame(np.random.rand(10, 4) * 5, columns=columns)
    predictions = model.predict(multiple_input)
    assert predictions.shape == (10,)
    
    # test wrong shape
    with pytest.raises(ValueError):
        wrong_df = pd.DataFrame([[5.1, 3.5, 1.4]], columns=['a', 'b', 'c'])
        model.predict(wrong_df)