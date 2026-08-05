from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================
# Input DTOs (Request)
# ==========================================

class DriverRaceData(BaseModel):
    """Represents a specific driver race data"""

    driver_ref: str = Field(..., description='Driver Reference Name (ex: \'max_verstappen\'')
    driver_age: float
    driver_momentum: float
    driver_track_affinity: float
    driver_dnf_rate: float

    position_qualifying: int
    grid: int
    q1_millis: float
    reached_q1: bool
    q2_millis: float
    reached_q2: bool
    q3_millis: float
    reached_q3: bool

    constructor_id: Optional[int] = None
    constructor_momentum: float
    constructor_track_affinity: float
    constructor_dnf_rate: float

    round: int
    circuit_id: Optional[int] = None
    circuit_dnf_rate: float


class RacePredictionRequest(BaseModel):
    """Main class used by the API to predict a race"""

    race_name: str = Field(..., description='Name or ID of the race')
    grid_data: List[DriverRaceData] = Field(..., description='List of all 20 drivers race data')


# ==========================================
# Output DTOs (Response)
# ==========================================

class DriverPrediction(BaseModel):
    """Represents the isolated driver prediction"""
    driver_ref: str
    predicted_score: float
    predicted_position: int


class RacePredictionResponse(BaseModel):
    """API response with the final grid ranked"""

    race: str
    status: str
    predictions: List[DriverPrediction]
