from datetime import date
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


class RacePredictionByDateRequest(BaseModel):
    """Main class used by the API to predict a race results using only a specific date"""
    race_date: date


class SeasonPredictionRequest(BaseModel):
    """Request used to predict a full season."""
    year: int
    use_current_results: bool = Field(
        default=False,
        description='Use official results for completed races and predict only future races',
    )
    current_form_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description='Weight applied to current-season points pace when use_current_results is enabled',
    )


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


class SeasonDriverStanding(BaseModel):
    standing_position: int
    driver_ref: str
    driver_name: str
    constructor_ref: str
    current_points: float
    predicted_points: float
    season_points: float
    constructor_points: float


class SeasonConstructorStanding(BaseModel):
    standing_position: int
    constructor_ref: str
    constructor_name: str
    current_points: float
    predicted_points: float
    constructor_points: float


class SeasonPredictionResponse(BaseModel):
    year: int
    status: str
    use_current_results: bool
    current_form_weight: float
    driver_standings: List[SeasonDriverStanding]
    constructor_standings: List[SeasonConstructorStanding]
