from pydantic import BaseModel

class TransactionInput(BaseModel):
    amount: float
    hour: int
    day: int
    category: int
