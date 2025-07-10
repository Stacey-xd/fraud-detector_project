from fastapi import FastAPI
from app.api import predict

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)

app.include_router(predict.router)