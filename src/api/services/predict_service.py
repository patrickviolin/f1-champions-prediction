from pathlib import Path

import pandas as pd
from xgboost import XGBRanker, XGBRegressor

from api.schemas.predict_dto import RacePredictionRequest, RacePredictionResponse, DriverPrediction


class PredictService:
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent.parent

        ranker_path = project_root / 'models' / 'xgb_ranker_model.json'
        regressor_path = project_root / 'models' / 'xgb_regressor_model.json'

        self.regressor_model = XGBRegressor()
        self.regressor_model.load_model(regressor_path)

        self.ranker_model = XGBRanker()
        self.ranker_model.load_model(ranker_path)

    def execute_prediction(self, request: RacePredictionRequest, model_type: str) -> RacePredictionResponse:
        """
        Execute prediction request: Converts the payload into DataFrame, predicts the score and returns the prediction result.
        :param model_type: regressor or ranker
        :param request:RacePredictionRequest
        :return:RacePredictionResponse
        """
        drivers_data = [driver.model_dump() for driver in request.grid_data]
        df_input = pd.DataFrame(drivers_data)

        driver_names = df_input['driver_ref'].copy()
        df_features = df_input.drop(columns='driver_ref')

        df_features = df_features.rename(columns={'constructor_id': 'constructorId', 'circuit_id': 'circuitId'})

        categorical_cols = ['constructorId', 'circuitId']
        for col in categorical_cols:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype('category')

        expected_order = [
            'driver_age', 'driver_momentum', 'position_qualifying', 'q1_millis',
            'reached_q1', 'q2_millis', 'reached_q2', 'q3_millis', 'reached_q3',
            'constructorId', 'constructor_momentum', 'round', 'circuitId', 'grid',
            'driver_track_affinity', 'constructor_track_affinity', 'constructor_dnf_rate',
            'driver_dnf_rate', 'circuit_dnf_rate'
        ]

        df_features = df_features[expected_order]

        if model_type.lower() == 'ranker':
            predictions = self.ranker_model.predict(df_features)
        else:
            predictions = self.regressor_model.predict(df_features)

        results = pd.DataFrame({
            'driver_ref': driver_names,
            'predicted_score': predictions
        })

        results = results.sort_values(by='predicted_score', ascending=False).reset_index(drop=True)
        results['predicted_position'] = results.index + 1

        predictions_list = []
        for _, row in results.iterrows():
            predictions_list.append(DriverPrediction(
                driver_ref=row['driver_ref'],
                predicted_score=float(row['predicted_score']),
                predicted_position=int(row['predicted_position'])
            ))

        return RacePredictionResponse(
            race=request.race_name,
            status='success',
            predictions=predictions_list
        )


predict_service = PredictService()
