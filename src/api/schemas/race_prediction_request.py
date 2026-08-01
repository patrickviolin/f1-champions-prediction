from typing import List

from pydantic import BaseModel, Field

from api.schemas.driver_race_data import DriverRaceData


class RacePredictionRequest(BaseModel):
    """Main class used by the API to predict a race"""

    race_name: str = Field(..., description='Name or ID of the race')
    grid_data: List[DriverRaceData] = Field(..., description='List of all 20 drivers')
