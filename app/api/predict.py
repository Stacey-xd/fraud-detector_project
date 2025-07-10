from fastapi import APIRouter
from app.schemas.input import TransactionData
from app.models.predict_model import predict_transaction

router = APIRouter()

@router.post("/predict")
def predict(data: TransactionData):
    probability = predict_transaction(data)
    return {"fraud_probability": probability}