from pathlib import Path

import pandas as pd
from xgboost import XGBRanker, XGBRegressor

from api.schemas.predict_dto import RacePredictionRequest, RacePredictionResponse, DriverPrediction, \
    RacePredictionByDateRequest, SeasonPredictionRequest, SeasonPredictionResponse, SeasonDriverStanding, \
    SeasonConstructorStanding
from application.services.f1db_data_to_ml_schema import F1DbDataToMlSchema
from application.services.f1db_season_prediction import F1DbSeasonPrediction


class PredictService:
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent.parent

        ranker_path = project_root / 'models' / 'xgb_ranker_model.json'
        regressor_path = project_root / 'models' / 'xgb_regressor_model.json'

        self.regressor_model = XGBRegressor()
        self.regressor_model.load_model(regressor_path)

        self.ranker_model = XGBRanker()
        self.ranker_model.load_model(ranker_path)

        self.f1db_data_to_ml_schema = F1DbDataToMlSchema.create_default()
        self.f1db_season_prediction = F1DbSeasonPrediction.create_default()

    def execute_season_prediction(self, request: SeasonPredictionRequest) -> SeasonPredictionResponse:
        driver_standings, constructor_standings = self.f1db_season_prediction.predict_season_standings(
            season_year=request.year,
            use_current_results=request.use_current_results,
        )

        return SeasonPredictionResponse(
            year=request.year,
            status='success',
            use_current_results=request.use_current_results,
            driver_standings=[
                SeasonDriverStanding(
                    standing_position=int(row['standing_position']),
                    driver_ref=str(row['driver_ref']),
                    driver_name=str(row['driver_name']),
                    constructor_ref=str(row['constructor_ref']),
                    current_points=int(row['current_points']),
                    predicted_points=int(row['predicted_points']),
                    season_points=int(row['season_points']),
                    constructor_points=int(row['constructor_points']),
                )
                for _, row in driver_standings.iterrows()
            ],
            constructor_standings=[
                SeasonConstructorStanding(
                    standing_position=int(row['standing_position']),
                    constructor_ref=str(row['constructor_ref']),
                    constructor_name=str(row['constructor_name']),
                    current_points=int(row['current_points']),
                    predicted_points=int(row['predicted_points']),
                    constructor_points=int(row['constructor_points']),
                )
                for _, row in constructor_standings.iterrows()
            ],
        )

    def execute_prediction_by_race_date(self, request: RacePredictionByDateRequest, model_type: str) -> RacePredictionResponse:
        """
        Execute the prediction request using only the race date. Since the ML model was trained with data until 2024,
        only races from 2025 onwards should be used.
        :param request: RacePredictionByDateRequest
        :param model_type: regressor or ranker
        :return:RacePredictionResponse
        """
        race_full_data = self.f1db_data_to_ml_schema.build_request(request.race_date)

        return self.execute_prediction(race_full_data, model_type)

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
                raw_list = [int(val) if pd.notna(val) else None for val in df_features[col]]

                df_features[col] = pd.Series(raw_list, index=df_features.index, dtype=object)

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
