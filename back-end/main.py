from fastapi import FastAPI
from data import HousingInput
from predictor import predict as run_prediction

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Housing Price Prediction API"}




@app.post("/predict")
def predict(data: HousingInput):

    prediction_data = data.model_dump()

    prediction = run_prediction(prediction_data)

    return prediction


def explain_prediction(housing_data: dict, prediction: dict):
    pass
