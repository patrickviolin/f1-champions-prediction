import pandas as pd
from pandas import DataFrame

from data.f1db_loader import F1DbRawData
from data.race_history_selector import RaceContext
from features.race_feature_engineering import RaceFeatureLookups
from utils import f1db_utils


class TargetRaceDataBuilder:
    def prepare_qualifying_data(self, qualifying: DataFrame) -> None:
        qualifying['reached_q1'] = qualifying['q1Millis'].notna()
        qualifying['reached_q2'] = qualifying['q2Millis'].notna()
        qualifying['reached_q3'] = qualifying['q3Millis'].notna()

        qualifying.rename(
            columns={
                'q1Millis': 'q1_millis',
                'q2Millis': 'q2_millis',
                'q3Millis': 'q3_millis',
                'positionDisplayOrder': 'position_qualifying',
            },
            inplace=True,
        )

        qualifying['q1_millis'] = qualifying['q1_millis'].fillna(0.0)
        qualifying['q2_millis'] = qualifying['q2_millis'].fillna(0.0)
        qualifying['q3_millis'] = qualifying['q3_millis'].fillna(0.0)

    def build(self, raw_data: F1DbRawData, context: RaceContext, feature_lookups: RaceFeatureLookups) -> DataFrame:
        mappings = f1db_utils.load_mappings()
        target_race_df = context.qualifying[context.qualifying['raceId'] == context.race_id_to_predict].copy()

        if target_race_df.empty:
            raise ValueError("There's no qualifying data for the chosen race. Is F1DB already updated with Saturday's qualifying data?"
            )

        target_race_df = self._merge_starting_grid(target_race_df, raw_data)
        target_race_df = self._merge_driver_age(target_race_df, raw_data)
        target_race_df = self._merge_driver_features(target_race_df, feature_lookups)
        target_race_df = self._merge_constructor_features(target_race_df, feature_lookups)

        self._fill_missing_features(target_race_df)
        target_race_df['circuit_dnf_rate'] = feature_lookups.circuit_dnf_rate
        target_race_df['circuitId'] = context.race_to_predict['circuitId']
        target_race_df['circuit_id_mapped'] = target_race_df['circuitId'].map(mappings['circuits'])
        target_race_df['constructor_id_mapped'] = target_race_df['constructorId'].map(mappings['constructors'])

        return target_race_df

    def _merge_starting_grid(self, target_race_df: DataFrame, raw_data: F1DbRawData) -> DataFrame:
        target_race_df = pd.merge(
            target_race_df,
            raw_data.starting_grid[['raceId', 'driverId', 'positionDisplayOrder']],
            on=['raceId', 'driverId'],
            how='left',
            validate='many_to_many',
        )
        target_race_df.rename(columns={'positionDisplayOrder': 'grid'}, inplace=True)
        return target_race_df

    def _merge_driver_age(self, target_race_df: DataFrame, raw_data: F1DbRawData) -> DataFrame:
        return pd.merge(
            target_race_df,
            raw_data.drivers[['id', 'driver_age']],
            left_on='driverId',
            right_on='id',
            how='left',
            validate='many_to_many',
        )

    def _merge_driver_features(
        self,
        target_race_df: DataFrame,
        feature_lookups: RaceFeatureLookups,
    ) -> DataFrame:
        target_race_df = pd.merge(
            target_race_df,
            feature_lookups.driver_momentum_lookup,
            on='driverId',
            how='left',
            validate='many_to_many',
        )
        target_race_df = pd.merge(
            target_race_df,
            feature_lookups.driver_track_lookup,
            on='driverId',
            how='left',
            validate='many_to_many',
        )
        return pd.merge(
            target_race_df,
            feature_lookups.driver_dnf_lookup,
            on='driverId',
            how='left',
            validate='many_to_many',
        )

    def _merge_constructor_features(
        self,
        target_race_df: DataFrame,
        feature_lookups: RaceFeatureLookups,
    ) -> DataFrame:
        target_race_df = pd.merge(
            target_race_df,
            feature_lookups.constructor_momentum_lookup,
            on='constructorId',
            how='left',
            validate='many_to_many',
        )
        target_race_df = pd.merge(
            target_race_df,
            feature_lookups.constructor_track_lookup,
            on='constructorId',
            how='left',
            validate='many_to_many',
        )
        return pd.merge(
            target_race_df,
            feature_lookups.constructor_dnf_lookup,
            on='constructorId',
            how='left',
            validate='many_to_many',
        )

    def _fill_missing_features(self, target_race_df: DataFrame) -> None:
        cols_fill = [
            'driver_momentum',
            'driver_track_affinity',
            'driver_dnf_rate',
            'constructor_momentum',
            'constructor_track_affinity',
            'constructor_dnf_rate',
        ]
        target_race_df[cols_fill] = target_race_df[cols_fill].fillna(0.0)
