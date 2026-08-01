from pydantic import BaseModel


class DriverPrediction(BaseModel):
    """Represents the isolated driver prediction"""
    driver_ref: str
    predicted_score: float
    predicted_position: int
