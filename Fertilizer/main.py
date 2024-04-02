import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the pickled model
with open("fertilizer_predictor.pkl", "rb") as f:
    fertilizer_predictor = pickle.load(f)

# Define the Pydantic model for input data
class Fertilizer(BaseModel):
    N: float
    P: float
    Ph: float
    temp: float
    hum: float
    mois: float
    soil: float
    crop: float

@app.get('/')
def root():
    return {'message': 'Welcome to the Fertilizer Prediction API'}

@app.post('/predict', response_model=dict)
def predict_fertilizer(data: Fertilizer):
    """Route to make predictions using the model."""
    try:
        # Access data attributes directly from the Pydantic model
        prediction = fertilizer_predictor.predict([[data.temp, data.hum, data.mois, data.soil, data.crop, data.N, data.P,data.Ph]])
        return {'prediction': prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)