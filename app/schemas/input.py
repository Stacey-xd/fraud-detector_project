from pydantic import BaseModel

class TransactionData(BaseModel):
    V1: float
    V2: float
    V3: float
    Amount: float
    Time: float