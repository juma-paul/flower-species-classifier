import pytest
from fastapi.testclient import TestClient
from src.api.app import app

@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Ensure model is loaded before tests"""
    from src.api.app import model
    if model is None:
        import joblib
        import src.api.app as app_module
        app_module.model = joblib.load('models/knn_model.pkl')

client = TestClient(app)

# Unit Tests
def test_root_endpoint():
    """Test the root endpoint returns API information"""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json() or "Info" in response.json()

def test_predict_endpoint_valid_input():
    """Test single prediction with valid input"""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1, 2]

def test_predict_endpoint_invalid_input():
    """Test single prediction with missing fields"""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_endpoint_wrong_type():
    """Test single prediction with wrong data type"""
    payload = {
        "sepal_length": "not_a_number",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_batch_predict_endpoint_valid_input():
    """Test batch prediction with valid input"""
    payload = {
        "feature_list": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.7,
                "sepal_width": 3.0,
                "petal_length": 5.2,
                "petal_width": 2.3
            }
        ]
    }
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert all(pred in [0, 1, 2] for pred in predictions)

def test_batch_predict_endpoint_empty_list():
    """Test batch prediction with empty list"""
    payload = {"feature_list": []}
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    assert response.json() == []

# Integration Tests
def test_api_full_workflow():
    """Integration test: full prediction workflow"""
    # check API is running
    root_response = client.get("/")
    assert root_response.status_code == 200
    
    # make single prediction
    single_payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 5.1,
        "petal_width": 1.9
    }
    single_response = client.post("/predict", json=single_payload)
    assert single_response.status_code == 200
    
    # make batch prediction
    batch_payload = {
        "feature_list": [single_payload, single_payload]
    }
    batch_response = client.post("/predict-batch", json=batch_payload)
    assert batch_response.status_code == 200
    assert len(batch_response.json()) == 2
