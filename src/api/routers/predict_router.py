from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from api.schemas.predict_dto import RacePredictionRequest, RacePredictionResponse
from api.services.predict_service import predict_service

router = APIRouter(
    prefix="/predict",
    tags=["Race Prediction"]
)


@router.post('/race', response_model=RacePredictionResponse)
async def predict_race(
        request: RacePredictionRequest,
        model_type: Annotated[str, Query('ranker', description="Choose a model: 'ranker'  or 'regressor'")]
):
    """
    Receive grid start data, driver and circuit stats
    Returns the predicted final race result according to the score
    :param model_type:ranker or regressor
    :param request:RacePredictionRequest
    :return:RacePredictionResponse
    """
    try:
        return predict_service.execute_prediction(request, model_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal error when processing request. Error: {e}')
