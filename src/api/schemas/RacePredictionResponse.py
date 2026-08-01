from typing import List

from pydantic import BaseModel

from api.schemas.driver_prediction import DriverPrediction


class RacePredictionResponse(BaseModel):
    """API response with the final grid ranked"""

    race: str
    status: str
    predictions: List[DriverPrediction]
