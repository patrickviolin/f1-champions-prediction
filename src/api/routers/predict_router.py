from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from api.schemas.predict_dto import RacePredictionRequest, RacePredictionResponse, RacePredictionByDateRequest
from application.services.predict_service import predict_service

router = APIRouter(
    prefix="/predict",
    tags=["Race Prediction"]
)


@router.post('/race', response_model=RacePredictionResponse,
             responses={500: {"description": "Internal server error when predicting"}})
async def predict_race(
        request: RacePredictionRequest,
        model_type: Annotated[str, Query(description="Choose a model: 'ranker'  or 'regressor'")] = 'ranker'
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


@router.post('/race-by-date', response_model=RacePredictionResponse,
             responses={500: {"description": "Internal server error when predicting"}})
async def predict_race(
        request: RacePredictionByDateRequest,
        model_type: Annotated[str, Query(description="Choose a model: 'ranker'  or 'regressor'")] = 'ranker'
):
    """
    Receive grid start data, driver and circuit stats
    Returns the predicted final race result according to the score
    :param model_type:ranker or regressor
    :param request:RacePredictionRequest
    :return:RacePredictionResponse
    """
    try:
        return predict_service.execute_prediction_by_race_date(request, model_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal error when processing request. Error: {e}')
