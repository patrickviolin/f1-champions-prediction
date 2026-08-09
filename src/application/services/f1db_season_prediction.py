from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series
from xgboost import XGBRanker

from application.services.f1db_data_to_ml_schema import F1DbDataToMlSchema
from data.f1db_loader import F1DbDataLoader, F1DbRawData
from data.race_history_selector import RaceContext, RaceHistorySelector
from features.current_season_feature_engineering import build_f1db_current_season_features
from features.race_feature_engineering import RaceFeatureEngineering, RaceFeatureLookups
from ml.pre_qualifying_features import PRE_QUALIFYING_CATEGORICAL_COLUMNS, PRE_QUALIFYING_FEATURE_ORDER
from utils import f1db_utils

F1_POINTS_BY_POSITION = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

DEFAULT_CURRENT_FORM_WEIGHT = 0.0
INACTIVE_PREVIOUS_SEASON_SCORE_PENALTY = 0.20
INACTIVE_WITH_UNKNOWN_CONSTRUCTOR_SCORE_PENALTY = 0.60
PREVIOUS_SEASON_DRIVER_SCORE_WEIGHT = 1.40
PREVIOUS_SEASON_CONSTRUCTOR_SCORE_WEIGHT = 0.80
PREVIOUS_SEASON_POSITION_BONUSES = {
    1: 0.60,
    2: 0.30,
    3: 0.20,
}

POST_QUALIFYING_FEATURE_ORDER = [
    'driver_age',
    'driver_momentum',
    'position_qualifying',
    'q1_millis',
    'reached_q1',
    'q2_millis',
    'reached_q2',
    'q3_millis',
    'reached_q3',
    'constructorId',
    'constructor_momentum',
    'round',
    'circuitId',
    'grid',
    'driver_track_affinity',
    'constructor_track_affinity',
    'constructor_dnf_rate',
    'driver_dnf_rate',
    'circuit_dnf_rate',
    'current_season_points_per_race',
    'current_season_avg_finish',
    'current_season_podium_rate',
    'current_season_q3_rate',
    'current_constructor_points_per_race',
    'last_3_current_season_avg_finish',
]

POST_QUALIFYING_CATEGORICAL_COLUMNS = ['constructorId', 'circuitId']


def _with_driver_age(raw_data: F1DbRawData, race_date: date) -> F1DbRawData:
    drivers = raw_data.drivers.copy()
    drivers['driver_age'] = (pd.to_datetime(race_date) - drivers['dateOfBirth']) / pd.Timedelta(days=365.25)
    return replace(raw_data, drivers=drivers)


def _build_past_race_results(context: RaceContext) -> DataFrame:
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


def _driver_entered_round(rounds: object, round_to_predict: int) -> bool:
    if pd.isna(rounds):
        return False

    return str(round_to_predict) in str(rounds).split(';')


def _merge_feature_lookups(target_race_df: DataFrame, feature_lookups: RaceFeatureLookups) -> DataFrame:
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
    target_race_df = pd.merge(
        target_race_df,
        feature_lookups.driver_dnf_lookup,
        on='driverId',
        how='left',
        validate='many_to_many',
    )
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


def _entries_for_race(season_entries: DataFrame, race_to_predict: Series) -> DataFrame:
    round_to_predict = int(race_to_predict['round'])
    entries = season_entries[
        (season_entries['year'] == int(race_to_predict['year']))
        & (~season_entries['testDriver'])
        ].copy()

    entries_for_round = entries[
        entries['rounds'].apply(lambda rounds: _driver_entered_round(rounds, round_to_predict))
    ][['driverId', 'constructorId']]

    if not entries_for_round.empty:
        return entries_for_round

    return entries[['driverId', 'constructorId']].drop_duplicates()


def _build_season_entry_features(
        raw_data: F1DbRawData,
        context: RaceContext,
        feature_lookups: RaceFeatureLookups,
) -> DataFrame:
    mappings = f1db_utils.load_mappings()
    entries = _entries_for_race(raw_data.season_entrants_drivers, context.race_to_predict)

    target_race_df = pd.merge(
        entries,
        raw_data.drivers[['id', 'driver_age']],
        left_on='driverId',
        right_on='id',
        how='left',
        validate='many_to_one',
    )
    target_race_df = _merge_feature_lookups(target_race_df, feature_lookups)

    cols_fill = [
        'driver_momentum',
        'driver_track_affinity',
        'driver_dnf_rate',
        'constructor_momentum',
        'constructor_track_affinity',
        'constructor_dnf_rate',
    ]
    target_race_df[cols_fill] = target_race_df[cols_fill].fillna(0.0)
    target_race_df['round'] = int(context.race_to_predict['round'])
    target_race_df['circuit_dnf_rate'] = feature_lookups.circuit_dnf_rate
    target_race_df['circuitId'] = context.race_to_predict['circuitId']
    target_race_df['constructor_ref'] = target_race_df['constructorId']
    target_race_df['constructorId'] = target_race_df['constructorId'].map(mappings['constructors'])
    target_race_df['circuitId'] = target_race_df['circuitId'].map(mappings['circuits'])

    return target_race_df


def _has_qualifying_data(race: Series, raw_data: F1DbRawData) -> bool:
    return not raw_data.qualifying[raw_data.qualifying['raceId'] == race['id']].empty


def _completed_race_ids(season_year: int, raw_data: F1DbRawData) -> set:
    season_race_ids = raw_data.races[raw_data.races['year'] == season_year]['id']
    race_results = raw_data.race_results[raw_data.race_results['raceId'].isin(season_race_ids)]
    return set(race_results['raceId'].unique())


def _build_driver_standings(race_predictions: DataFrame, raw_data: F1DbRawData) -> DataFrame:
    driver_standings = (
        race_predictions.groupby(['driver_ref', 'constructor_ref'], as_index=False)['points']
        .sum()
        .rename(columns={'points': 'season_points'})
        .sort_values(by='season_points', ascending=False)
        .reset_index(drop=True)
    )
    driver_standings['standing_position'] = driver_standings.index + 1
    driver_standings['current_points'] = 0.0
    driver_standings['predicted_points'] = driver_standings['season_points']

    return pd.merge(
        driver_standings,
        raw_data.drivers[['id', 'name']],
        left_on='driver_ref',
        right_on='id',
        how='left',
        validate='many_to_one',
    ).rename(columns={'name': 'driver_name'}).drop(columns='id')


def _build_constructor_standings(race_predictions: DataFrame, raw_data: F1DbRawData) -> DataFrame:
    constructor_standings = (
        race_predictions.groupby('constructor_ref', as_index=False)['points']
        .sum()
        .rename(columns={'points': 'constructor_points'})
        .sort_values(by='constructor_points', ascending=False)
        .reset_index(drop=True)
    )
    constructor_standings['standing_position'] = constructor_standings.index + 1
    constructor_standings['current_points'] = 0.0
    constructor_standings['predicted_points'] = constructor_standings['constructor_points']

    return pd.merge(
        constructor_standings,
        raw_data.constructors[['id', 'name']],
        left_on='constructor_ref',
        right_on='id',
        how='left',
        validate='many_to_one',
    ).rename(columns={'name': 'constructor_name'}).drop(columns='id')


def _latest_driver_standings(season_year: int, raw_data: F1DbRawData) -> DataFrame:
    standings = raw_data.race_driver_standings[raw_data.race_driver_standings['year'] == season_year].copy()

    if standings.empty:
        return DataFrame(columns=['driver_ref', 'constructor_ref', 'current_points'])

    standings = standings[standings['round'] == standings['round'].max()]
    constructor_lookup = _driver_constructor_lookup(season_year, raw_data)
    standings = standings.rename(columns={'driverId': 'driver_ref', 'points': 'current_points'})
    standings['constructor_ref'] = standings['driver_ref'].map(constructor_lookup)

    return standings[['driver_ref', 'constructor_ref', 'current_points']]


def _latest_constructor_standings(season_year: int, raw_data: F1DbRawData) -> DataFrame:
    standings = raw_data.race_constructor_standings[
        raw_data.race_constructor_standings['year'] == season_year
        ].copy()

    if standings.empty:
        return DataFrame(columns=['constructor_ref', 'current_points'])

    standings = standings[standings['round'] == standings['round'].max()]
    return standings.rename(
        columns={'constructorId': 'constructor_ref', 'points': 'current_points'}
    )[['constructor_ref', 'current_points']]


def _driver_constructor_lookup(season_year: int, raw_data: F1DbRawData) -> Series:
    season_results = raw_data.race_results[raw_data.race_results['year'] == season_year].copy()

    if not season_results.empty:
        return (
            season_results.sort_values(by='round')
            .drop_duplicates('driverId', keep='last')
            .set_index('driverId')['constructorId']
        )

    season_entries = raw_data.season_entrants_drivers[
        (raw_data.season_entrants_drivers['year'] == season_year)
        & (~raw_data.season_entrants_drivers['testDriver'])
        ]
    return season_entries.drop_duplicates('driverId').set_index('driverId')['constructorId']


def _future_driver_points(future_predictions: DataFrame) -> DataFrame:
    if future_predictions.empty:
        return DataFrame(columns=['driver_ref', 'future_constructor_ref', 'future_points'])

    return (
        future_predictions.groupby('driver_ref', as_index=False)
        .agg(future_points=('points', 'sum'), future_constructor_ref=('constructor_ref', 'last'))
    )


def _future_constructor_points(future_predictions: DataFrame) -> DataFrame | None | Series[Any]:
    if future_predictions.empty:
        return DataFrame(columns=['constructor_ref', 'future_points'])

    return (
        future_predictions.groupby('constructor_ref', as_index=False)['points']
        .sum()
        .rename(columns={'points': 'future_points'})
    )


def _combine_driver_standings(
        current_driver_standings: DataFrame,
        future_predictions: DataFrame,
        raw_data: F1DbRawData,
        completed_races_count: int,
        current_form_weight: float,
) -> DataFrame:
    standings = pd.merge(
        current_driver_standings,
        _future_driver_points(future_predictions),
        on='driver_ref',
        how='outer',
        validate='many_to_many'
    )
    setup_current_and_future_points(completed_races_count, current_form_weight, future_predictions, standings)

    standings['constructor_ref'] = standings['constructor_ref'].fillna(standings['future_constructor_ref'])
    standings['season_points'] = standings['current_points'] + standings['future_points']
    standings = standings.sort_values(by='season_points', ascending=False).reset_index(drop=True)
    standings['standing_position'] = standings.index + 1

    return pd.merge(
        standings[
            [
                'standing_position',
                'driver_ref',
                'constructor_ref',
                'current_points',
                'future_points',
                'season_points',
            ]
        ],
        raw_data.drivers[['id', 'name']],
        left_on='driver_ref',
        right_on='id',
        how='left',
        validate='many_to_one',
    ).rename(columns={'future_points': 'predicted_points', 'name': 'driver_name'}).drop(columns='id')


def _combine_constructor_standings(
        current_constructor_standings: DataFrame,
        future_predictions: DataFrame,
        raw_data: F1DbRawData,
        completed_races_count: int,
        current_form_weight: float,
) -> DataFrame:
    standings = pd.merge(
        current_constructor_standings,
        _future_constructor_points(future_predictions),
        on='constructor_ref',
        how='outer',
        validate='many_to_many'
    )
    setup_current_and_future_points(completed_races_count, current_form_weight, future_predictions, standings)
    standings['constructor_points'] = standings['current_points'] + standings['future_points']
    standings = standings.sort_values(by='constructor_points', ascending=False).reset_index(drop=True)
    standings['standing_position'] = standings.index + 1

    return pd.merge(
        standings[
            [
                'standing_position',
                'constructor_ref',
                'current_points',
                'future_points',
                'constructor_points',
            ]
        ],
        raw_data.constructors[['id', 'name']],
        left_on='constructor_ref',
        right_on='id',
        how='left',
        validate='many_to_one',
    ).rename(columns={'future_points': 'predicted_points', 'name': 'constructor_name'}).drop(columns='id')


def setup_current_and_future_points(completed_races_count: int, current_form_weight: float,
                                    future_predictions: DataFrame, standings: DataFrame):
    standings['current_points'] = standings['current_points'].fillna(0.0)
    standings['future_points'] = standings['future_points'].fillna(0.0)
    standings['future_points'] = _apply_current_form_adjustment(
        current_points=standings['current_points'],
        ml_future_points=standings['future_points'],
        future_races_count=_future_races_count(future_predictions),
        completed_races_count=completed_races_count,
        current_form_weight=current_form_weight,
    )


def _future_races_count(future_predictions: DataFrame) -> int:
    if future_predictions.empty:
        return 0

    return int(future_predictions['race_id'].nunique())


def _apply_current_form_adjustment(
        current_points: Series,
        ml_future_points: Series,
        future_races_count: int,
        completed_races_count: int,
        current_form_weight: float,
) -> Series:
    if completed_races_count == 0 or future_races_count == 0 or current_form_weight == 0:
        return ml_future_points

    current_pace_projection = (current_points / completed_races_count) * future_races_count
    return (ml_future_points * (1 - current_form_weight)) + (current_pace_projection * current_form_weight)


def _build_prediction_results(
        race: Series,
        driver_refs: Series,
        constructor_refs: Series,
        scores: object,
) -> DataFrame:
    results = pd.DataFrame(
        {
            'race_id': race['id'],
            'race_name': race['officialName'],
            'round': int(race['round']),
            'driver_ref': driver_refs,
            'constructor_ref': constructor_refs,
            'predicted_score': scores,
        }
    )
    results = results.sort_values(by='predicted_score', ascending=False).reset_index(drop=True)
    results['predicted_position'] = results.index + 1

    return results


class F1DbSeasonPrediction:
    def __init__(
            self,
            loader: F1DbDataLoader,
            history_selector: RaceHistorySelector,
            feature_engineer: RaceFeatureEngineering,
            pre_qualifying_model: XGBRanker,
            post_qualifying_model: XGBRanker,
            f1db_data_to_ml_schema: F1DbDataToMlSchema,
            categorical_categories: dict[str, list[int]],
    ):
        self.loader = loader
        self.history_selector = history_selector
        self.feature_engineer = feature_engineer
        self.pre_qualifying_model = pre_qualifying_model
        self.post_qualifying_model = post_qualifying_model
        self.f1db_data_to_ml_schema = f1db_data_to_ml_schema
        self.categorical_categories = categorical_categories

    @classmethod
    def create_default(cls) -> 'F1DbSeasonPrediction':
        project_root = Path(__file__).parent.parent.parent.parent
        pre_qualifying_model = XGBRanker()
        pre_qualifying_model.load_model(project_root / 'models' / 'xgb_pre_qualifying_ranker_model.json')

        post_qualifying_model = XGBRanker()
        post_qualifying_model.load_model(project_root / 'models' / 'xgb_ranker_model.json')

        return cls(
            loader=F1DbDataLoader(project_root),
            history_selector=RaceHistorySelector(),
            feature_engineer=RaceFeatureEngineering(),
            pre_qualifying_model=pre_qualifying_model,
            post_qualifying_model=post_qualifying_model,
            f1db_data_to_ml_schema=F1DbDataToMlSchema.create_default(),
            categorical_categories=cls._load_training_categories(
                project_root / 'train_and_test_data' / '03_processed'
            ),
        )

    @staticmethod
    def _load_training_categories(data_dir: Path) -> dict[str, list[int]]:
        x_train = pd.read_csv(data_dir / 'X_train.csv')
        x_test = pd.read_csv(data_dir / 'X_test.csv')

        categories = {}
        for col in PRE_QUALIFYING_CATEGORICAL_COLUMNS:
            values = pd.concat([x_train[col], x_test[col]]).dropna().unique()
            categories[col] = [int(value) for value in values]

        return categories

    def predict_season_standings(
            self,
            season_year: int,
            use_current_results: bool = False,
            current_form_weight: float = DEFAULT_CURRENT_FORM_WEIGHT,
    ) -> tuple[DataFrame, DataFrame]:
        raw_data = self.loader.load()

        if use_current_results:
            return self._predict_season_standings_from_current_results(
                season_year=season_year,
                raw_data=raw_data,
                current_form_weight=current_form_weight,
            )

        race_predictions = self._predict_season_races(
            season_year=season_year,
            raw_data=raw_data,
            use_best_available_model=False,
        )

        if race_predictions.empty:
            return DataFrame(), DataFrame()

        if 'points' not in race_predictions.columns:
            race_predictions['points'] = pd.NA

        race_predictions['points'] = race_predictions['points'].fillna(
            race_predictions['predicted_position'].map(F1_POINTS_BY_POSITION)
        ).fillna(0).astype(float)

        driver_standings = _build_driver_standings(race_predictions, raw_data)
        constructor_standings = _build_constructor_standings(race_predictions, raw_data)
        driver_standings = pd.merge(
            driver_standings,
            constructor_standings[['constructor_ref', 'constructor_points']],
            on='constructor_ref',
            how='left',
            validate='many_to_one',
        )

        return driver_standings, constructor_standings

    def _predict_season_standings_from_current_results(
            self,
            season_year: int,
            raw_data: F1DbRawData,
            current_form_weight: float,
    ) -> tuple[DataFrame, DataFrame]:
        completed_races_count = len(_completed_race_ids(season_year, raw_data))
        future_predictions = self._predict_season_races(
            season_year=season_year,
            raw_data=raw_data,
            use_best_available_model=True,
        )

        if not future_predictions.empty:
            future_predictions['points'] = (
                future_predictions['predicted_position']
                .map(F1_POINTS_BY_POSITION)
                .fillna(0)
                .astype(float)
            )

        driver_standings = _combine_driver_standings(
            _latest_driver_standings(season_year, raw_data),
            future_predictions,
            raw_data,
            completed_races_count,
            current_form_weight,
        )
        constructor_standings = _combine_constructor_standings(
            _latest_constructor_standings(season_year, raw_data),
            future_predictions,
            raw_data,
            completed_races_count,
            current_form_weight,
        )
        driver_standings = pd.merge(
            driver_standings,
            constructor_standings[['constructor_ref', 'constructor_points']],
            on='constructor_ref',
            how='left',
            validate='many_to_one',
        )

        return driver_standings, constructor_standings

    def predict_season(self, season_year: int) -> DataFrame:
        raw_data = self.loader.load()
        return self._predict_season_races(season_year, raw_data)

    def predict_race_pre_qualifying(self, race_date: date) -> DataFrame:
        raw_data = self.loader.load()
        race = raw_data.races[raw_data.races['date'] == pd.to_datetime(race_date)].iloc[0]
        race_features = self._build_pre_qualifying_race_features(
            raw_data=raw_data,
            race_date=race_date,
            race_results=raw_data.race_results,
            qualifying=raw_data.qualifying,
        )
        return self._predict_pre_qualifying_race(race, race_features)

    def _predict_season_races(
            self,
            season_year: int,
            raw_data: F1DbRawData,
            use_best_available_model: bool = False,
    ) -> DataFrame:
        season_races = raw_data.races[raw_data.races['year'] == season_year].sort_values(by='round')

        if use_best_available_model:
            completed_race_ids = _completed_race_ids(season_year, raw_data)
            season_races = season_races[~season_races['id'].isin(completed_race_ids)]

        predictions = []
        simulated_race_results = self._initial_simulated_race_results(
            season_year=season_year,
            raw_data=raw_data,
            use_best_available_model=use_best_available_model,
        )
        simulated_qualifying = self._initial_simulated_qualifying(
            season_year=season_year,
            raw_data=raw_data,
            use_best_available_model=use_best_available_model,
        )

        for _, race in season_races.iterrows():
            if use_best_available_model and _has_qualifying_data(race, raw_data):
                race_predictions = self._predict_post_qualifying_race(race, raw_data)
            else:
                race_date = pd.to_datetime(race['date']).date()
                race_features = self._build_pre_qualifying_race_features(
                    raw_data=raw_data,
                    race_date=race_date,
                    race_results=simulated_race_results,
                    qualifying=simulated_qualifying,
                )
                race_predictions = self._predict_pre_qualifying_race(race, race_features)

                if not use_best_available_model:
                    race_predictions = self._apply_preseason_calibration(
                        season_year=season_year,
                        raw_data=raw_data,
                        race_predictions=race_predictions,
                    )

            predictions.append(race_predictions)

            if use_best_available_model:
                simulated_race_results = pd.concat(
                    [
                        simulated_race_results,
                        self._prediction_results_to_race_results(race, race_predictions),
                    ],
                    ignore_index=True,
                )

        if not predictions:
            return DataFrame()

        return pd.concat(predictions, ignore_index=True)

    def _initial_simulated_race_results(
            self,
            season_year: int,
            raw_data: F1DbRawData,
            use_best_available_model: bool,
    ) -> DataFrame:
        if use_best_available_model:
            return raw_data.race_results.copy()

        return raw_data.race_results[raw_data.race_results['year'] != season_year].copy()

    def _initial_simulated_qualifying(
            self,
            season_year: int,
            raw_data: F1DbRawData,
            use_best_available_model: bool,
    ) -> DataFrame:
        if use_best_available_model:
            return raw_data.qualifying.copy()

        return raw_data.qualifying[raw_data.qualifying['year'] != season_year].copy()

    def _prediction_results_to_race_results(self, race: Series, race_predictions: DataFrame) -> DataFrame:
        race_results = race_predictions.copy()
        race_results['raceId'] = race['id']
        race_results['year'] = int(race['year'])
        race_results['round'] = int(race['round'])
        race_results['driverId'] = race_results['driver_ref']
        race_results['constructorId'] = race_results['constructor_ref']
        race_results['positionDisplayOrder'] = race_results['predicted_position']
        race_results['positionNumber'] = race_results['predicted_position']
        race_results['points'] = race_results['predicted_position'].map(F1_POINTS_BY_POSITION).fillna(0.0)

        return race_results[
            [
                'raceId',
                'year',
                'round',
                'driverId',
                'constructorId',
                'positionDisplayOrder',
                'positionNumber',
                'points',
            ]
        ]

    def _build_pre_qualifying_race_features(
            self,
            raw_data: F1DbRawData,
            race_date: date,
            race_results: DataFrame,
            qualifying: DataFrame,
    ) -> DataFrame:
        raw_data = _with_driver_age(raw_data, race_date)
        raw_data = replace(raw_data, race_results=race_results, qualifying=qualifying)
        context = self.history_selector.select(raw_data, race_date)
        race_results_past = _build_past_race_results(context)
        feature_lookups = self.feature_engineer.build_features(race_results_past, context.race_to_predict)

        race_features = _build_season_entry_features(raw_data, context, feature_lookups)
        return build_f1db_current_season_features(
            race_features=race_features,
            race_results=race_results,
            qualifying=qualifying,
            race=context.race_to_predict,
        )

    def _predict_post_qualifying_race(self, race: Series, raw_data: F1DbRawData) -> DataFrame:
        request = self.f1db_data_to_ml_schema.build_request(pd.to_datetime(race['date']).date())
        drivers_data = [driver.model_dump() for driver in request.grid_data]
        df_input = pd.DataFrame(drivers_data)

        driver_refs = df_input['driver_ref'].copy()
        model_features = df_input.drop(columns='driver_ref')
        model_features = model_features.rename(columns={'constructor_id': 'constructorId', 'circuit_id': 'circuitId'})
        model_features = model_features[POST_QUALIFYING_FEATURE_ORDER].copy()
        model_features = self._apply_known_categories(model_features, POST_QUALIFYING_CATEGORICAL_COLUMNS)

        scores = self.post_qualifying_model.predict(model_features)
        constructor_refs = self._constructor_refs_for_race(raw_data, race['id'], driver_refs)

        return _build_prediction_results(
            race=race,
            driver_refs=driver_refs,
            constructor_refs=constructor_refs,
            scores=scores,
        )

    def _apply_preseason_calibration(
            self,
            season_year: int,
            raw_data: F1DbRawData,
            race_predictions: DataFrame,
    ) -> DataFrame:
        previous_season_driver_races = self._previous_season_driver_race_counts(season_year, raw_data)
        previous_driver_strength = self._previous_season_driver_strength(season_year, raw_data)
        previous_constructor_strength = self._previous_season_constructor_strength(season_year, raw_data)
        known_constructors = set(f1db_utils.load_mappings()['constructors'])
        race_predictions = race_predictions.copy()

        race_predictions['penalty'] = race_predictions.apply(
            lambda row: self._preseason_roster_penalty(
                driver_ref=row['driver_ref'],
                constructor_ref=row['constructor_ref'],
                previous_season_driver_races=previous_season_driver_races,
                known_constructors=known_constructors,
            ),
            axis=1,
        )
        race_predictions['previous_season_boost'] = race_predictions.apply(
            lambda row: self._previous_season_boost(
                driver_ref=row['driver_ref'],
                constructor_ref=row['constructor_ref'],
                previous_driver_strength=previous_driver_strength,
                previous_constructor_strength=previous_constructor_strength,
            ),
            axis=1,
        )
        race_predictions['predicted_score'] = (
                race_predictions['predicted_score']
                + race_predictions['previous_season_boost']
                - race_predictions['penalty']
        )
        race_predictions = race_predictions.drop(columns=['penalty', 'previous_season_boost'])
        race_predictions = race_predictions.sort_values(by='predicted_score', ascending=False).reset_index(drop=True)
        race_predictions['predicted_position'] = race_predictions.index + 1

        return race_predictions

    def _previous_season_driver_race_counts(self, season_year: int, raw_data: F1DbRawData) -> Series:
        previous_season_results = raw_data.race_results[raw_data.race_results['year'] == season_year - 1]
        return previous_season_results.groupby('driverId')['raceId'].nunique()

    def _preseason_roster_penalty(
            self,
            driver_ref: str,
            constructor_ref: str,
            previous_season_driver_races: Series,
            known_constructors: set[str],
    ) -> float:
        if previous_season_driver_races.get(driver_ref, 0) > 0:
            return 0.0

        if constructor_ref not in known_constructors:
            return INACTIVE_WITH_UNKNOWN_CONSTRUCTOR_SCORE_PENALTY

        return INACTIVE_PREVIOUS_SEASON_SCORE_PENALTY

    def _previous_season_driver_strength(self, season_year: int, raw_data: F1DbRawData) -> DataFrame:
        previous_standings = raw_data.race_driver_standings[
            raw_data.race_driver_standings['year'] == season_year - 1
            ].copy()

        if previous_standings.empty:
            return DataFrame(columns=['driver_ref', 'driver_score_boost'])

        previous_standings = previous_standings[previous_standings['round'] == previous_standings['round'].max()]
        max_points = previous_standings['points'].max() or 1
        previous_standings['driver_score_boost'] = (
                (previous_standings['points'] / max_points) * PREVIOUS_SEASON_DRIVER_SCORE_WEIGHT
                + previous_standings['positionDisplayOrder'].map(PREVIOUS_SEASON_POSITION_BONUSES).fillna(0.0)
        )

        return previous_standings.rename(columns={'driverId': 'driver_ref'})[
            ['driver_ref', 'driver_score_boost']
        ]

    def _previous_season_constructor_strength(self, season_year: int, raw_data: F1DbRawData) -> DataFrame:
        previous_standings = raw_data.race_constructor_standings[
            raw_data.race_constructor_standings['year'] == season_year - 1
            ].copy()

        if previous_standings.empty:
            return DataFrame(columns=['constructor_ref', 'constructor_score_boost'])

        previous_standings = previous_standings[previous_standings['round'] == previous_standings['round'].max()]
        max_points = previous_standings['points'].max() or 1
        previous_standings['constructor_score_boost'] = (
                                                                previous_standings['points'] / max_points
                                                        ) * PREVIOUS_SEASON_CONSTRUCTOR_SCORE_WEIGHT

        return previous_standings.rename(columns={'constructorId': 'constructor_ref'})[
            ['constructor_ref', 'constructor_score_boost']
        ]

    def _previous_season_boost(
            self,
            driver_ref: str,
            constructor_ref: str,
            previous_driver_strength: DataFrame,
            previous_constructor_strength: DataFrame,
    ) -> float:
        driver_boost = previous_driver_strength.loc[
            previous_driver_strength['driver_ref'] == driver_ref,
            'driver_score_boost',
        ]
        constructor_boost = previous_constructor_strength.loc[
            previous_constructor_strength['constructor_ref'] == constructor_ref,
            'constructor_score_boost',
        ]

        return float(
            (driver_boost.iloc[0] if not driver_boost.empty else 0.0)
            + (constructor_boost.iloc[0] if not constructor_boost.empty else 0.0)
        )

    def _constructor_refs_for_race(self, raw_data: F1DbRawData, race_id: object, driver_refs: Series) -> Series:
        qualifying = raw_data.qualifying[raw_data.qualifying['raceId'] == race_id][['driverId', 'constructorId']]
        constructor_lookup = qualifying.drop_duplicates('driverId').set_index('driverId')['constructorId']
        return driver_refs.map(constructor_lookup)

    def _predict_pre_qualifying_race(self, race: Series, race_features: DataFrame) -> DataFrame:
        driver_refs = race_features['driverId'].copy()
        constructor_refs = race_features['constructor_ref'].copy()
        model_features = race_features[PRE_QUALIFYING_FEATURE_ORDER].copy()
        model_features = self._apply_known_categories(model_features, PRE_QUALIFYING_CATEGORICAL_COLUMNS)

        scores = self.pre_qualifying_model.predict(model_features)

        return _build_prediction_results(
            race=race,
            driver_refs=driver_refs,
            constructor_refs=constructor_refs,
            scores=scores,
        )

    def _apply_known_categories(self, model_features: DataFrame, categorical_columns: list[str]) -> DataFrame:
        for col in categorical_columns:
            known_categories = self.categorical_categories[col]
            model_features[col] = pd.Series(
                [
                    int(value) if pd.notna(value) and int(value) in known_categories else None
                    for value in model_features[col]
                ],
                index=model_features.index,
                dtype=object,
            ).astype(pd.CategoricalDtype(categories=known_categories, ordered=False))

        return model_features
