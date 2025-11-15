import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Load model
model = None

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load('models/knn_model.pkl')
        print('Model loaded successfully!')
    except FileNotFoundError:
        print('Error: Model file not found!')
    except Exception as e:
        print(f'Error loading model: {e}')
    yield

    print('Shutting down...')

app = FastAPI(title='Iris Classifier API', lifespan=_lifespan)

# Pydantic model to validate inputs to the API
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisBatch(BaseModel):
    feature_list: list[IrisInput]

@app.get('/')
def root():
    return {
        "name": "Iris Classifier API",
        "version": "1.0.0",
        "description": "ML API for iris species classification",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict-batch",
            "docs": "/docs"
        }
    }


# Single data point API endpoint
@app.post('/predict')
def get_prediction(data: IrisInput):
    data_points = np.array([data.sepal_length, data.sepal_width, data.petal_length, data.petal_width])
    data_points = data_points.reshape(1, -1)

    # convert data points to a data frame
    data_df = pd.DataFrame(data_points, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = model.predict(data_df)
    print(prediction)

    return {'prediction': int(prediction[0])}

# Batch predictions API endpoint
@app.post('/predict-batch')
def get_batch_predictions(data: IrisBatch):
    predictions = []
    for data_point in data.feature_list:
        data_points = np.array([data_point.sepal_length, data_point.sepal_width, data_point.petal_length, data_point.petal_width])
        data_points = data_points.reshape(1, -1)

        # convert to dataframe
        data_df = pd.DataFrame(data_points, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        prediction = model.predict(data_df)
        print(prediction)

        predictions.append(int(prediction[0]))
    
    return predictions