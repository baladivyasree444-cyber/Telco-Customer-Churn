from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

model = joblib.load('telco_customer_churb.pkl')


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="This API predicts whether a customer will churn or not, based on their features.",
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int 
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float 
    TotalCharges: float 


@app.get("/")
def home():
    return {"message": "Telco Customer Churn Prediction API is Running"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([
        data.gender,
        data.SeniorCitizen,
        data.Partner,
        data.Dependents,
        data.tenure,
        data.PhoneService,
        data.MultipleLines,
        data.InternetService,
        data.OnlineSecurity,
        data.OnlineBackup,
        data.DeviceProtection,
        data.TechSupport,
        data.StreamingTV,
        data.StreamingMovies,
        data.Contract,
        data.PaperlessBilling,
        data.PaymentMethod,
        data.MonthlyCharges,
        data.TotalCharges
    ], index=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
              'MonthlyCharges', 'TotalCharges'
              ])


    input_df = pd.DataFrame([data.model_dump()])

    # Make predictions using the loaded model
    prediction = model.predict(input_df)

    # Return the prediction result
    return {"churn_prediction": int(prediction[0])} 