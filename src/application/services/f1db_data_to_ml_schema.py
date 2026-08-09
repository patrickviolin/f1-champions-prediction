from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from api.schemas.predict_dto import RacePredictionRequest
from data.f1db_loader import F1DbDataLoader, F1DbRawData
from data.race_history_selector import RaceContext, RaceHistorySelector
from data.target_race_data_builder import TargetRaceDataBuilder
from features.race_feature_engineering import RaceFeatureEngineering
from mappers.race_prediction_mapper import RacePredictionMapper


class F1DbDataToMlSchema:
    def __init__(
            self,
            loader: F1DbDataLoader,
            history_selector: RaceHistorySelector,
            feature_engineer: RaceFeatureEngineering,
            target_race_data_builder: TargetRaceDataBuilder,
            mapper: RacePredictionMapper
    ):
        self.loader = loader
        self.history_selector = history_selector
        self.feature_engineer = feature_engineer
        self.target_race_data_builder = target_race_data_builder
        self.mapper = mapper

    @classmethod
    def create_default(cls) -> F1DbDataToMlSchema:
        project_root = Path(__file__).parent.parent.parent.parent

        return cls(
            loader=F1DbDataLoader(project_root),
            history_selector=RaceHistorySelector(),
            feature_engineer=RaceFeatureEngineering(),
            target_race_data_builder=TargetRaceDataBuilder(),
            mapper=RacePredictionMapper()
        )

    def build_request(self, race_date: date) -> RacePredictionRequest:
        raw_data = self.loader.load()
        raw_data = self._with_driver_age(raw_data, race_date)

        context = self.history_selector.select(raw_data, race_date)
        self.target_race_data_builder.prepare_qualifying_data(context.qualifying)

        race_results_past = self._build_past_race_results(context)
        feature_lookups = self.feature_engineer.build_features(race_results_past, context.race_to_predict)
        target_race_df = self.target_race_data_builder.build(raw_data, context, feature_lookups)

        race_name = raw_data.races[raw_data.races['id'] == target_race_df['raceId'].iloc[0]]['officialName'].iloc[0]

        return self.mapper.to_request(target_race_df, race_name, context.round_to_predict)

    def _with_driver_age(self, raw_data: F1DbRawData, race_date: date) -> F1DbRawData:
        drivers = raw_data.drivers.copy()
        drivers['driver_age'] = (pd.to_datetime(race_date) - drivers['dateOfBirth']) / pd.Timedelta(days=365.25)
        return replace(raw_data, drivers=drivers)

    def _build_past_race_results(self, context: RaceContext) -> DataFrame:
        race_results_past = context.race_results[context.race_results['raceId'] < context.race_id_to_predict].copy()

        race_results_past = pd.merge(
            left=race_results_past,
            right=context.races_history[['id', 'date', 'circuitId']],
            left_on='raceId',
            right_on='id',
            how='inner',
            validate='many_to_many',
        )

        return race_results_past.sort_values(by='date')
