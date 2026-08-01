from pydantic import BaseModel, Field


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

    constructor_momentum: float
    constructor_track_affinity: float
    constructor_dnf_rate: float

    round: int
    circuit_id: int
    circuit_dnf_rate: float
