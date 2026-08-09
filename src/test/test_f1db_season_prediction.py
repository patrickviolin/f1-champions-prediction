from datetime import date
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from api.schemas.predict_dto import RacePredictionRequest, DriverRaceData
from application.services import f1db_season_prediction as season_prediction
from application.services.f1db_season_prediction import (
    INACTIVE_WITH_UNKNOWN_CONSTRUCTOR_SCORE_PENALTY,
    F1DbSeasonPrediction,
)
from data.f1db_loader import F1DbRawData
from data.race_history_selector import RaceContext
from features.race_feature_engineering import RaceFeatureLookups
from ml.pre_qualifying_features import PRE_QUALIFYING_FEATURE_ORDER


class FakeLoader:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def load(self):
        return self.raw_data


class FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.received_features = None

    def predict(self, features):
        self.received_features = features
        return self.scores


class FakeHistorySelector:
    def __init__(self, context):
        self.context = context

    def select(self, _raw_data, _race_date):
        return self.context


class FakeFeatureEngineer:
    def __init__(self, lookups):
        self.lookups = lookups
        self.received_race_results_past = None

    def build_features(self, race_results_past, _race_to_predict):
        self.received_race_results_past = race_results_past
        return self.lookups


class FakeF1DbDataToMlSchema:
    def __init__(self):
        self.received_race_date = None

    def build_request(self, race_date):
        self.received_race_date = race_date

        return RacePredictionRequest(
            race_name='Test Grand Prix',
            grid_data=[
                DriverRaceData(
                    driver_ref='driver-a',
                    driver_age=25.0,
                    driver_momentum=1.0,
                    current_season_points_per_race=0.0,
                    current_season_avg_finish=0.0,
                    current_season_podium_rate=0.0,
                    current_season_q3_rate=0.0,
                    driver_track_affinity=1.0,
                    driver_dnf_rate=0.0,
                    position_qualifying=1,
                    grid=1,
                    q1_millis=90000.0,
                    reached_q1=True,
                    q2_millis=89000.0,
                    reached_q2=True,
                    q3_millis=88000.0,
                    reached_q3=True,
                    constructor_id=1,
                    constructor_momentum=1.0,
                    current_constructor_points_per_race=0.0,
                    constructor_track_affinity=1.0,
                    constructor_dnf_rate=0.0,
                    round=1,
                    circuit_id=1,
                    last_3_current_season_avg_finish=0.0,
                    circuit_dnf_rate=0.0,
                )
            ],
        )


class F1DbSeasonPredictionTest(TestCase):
    def setUp(self):
        self.raw_data = F1DbRawData(
            constructors=pd.DataFrame(
                [
                    {'id': 'mclaren', 'name': 'McLaren'},
                    {'id': 'cadillac', 'name': 'Cadillac'},
                ]
            ),
            drivers=pd.DataFrame(
                [
                    {'id': 'lando-norris', 'name': 'Lando Norris', 'dateOfBirth': pd.Timestamp('1999-11-13')},
                    {'id': 'sergio-perez', 'name': 'Sergio Perez', 'dateOfBirth': pd.Timestamp('1990-01-26')},
                    {'id': 'driver-a', 'name': 'Driver A', 'dateOfBirth': pd.Timestamp('2000-01-01')},
                ]
            ),
            qualifying=pd.DataFrame(
                [
                    {
                        'raceId': 2,
                        'year': 2026,
                        'round': 1,
                        'driverId': 'driver-a',
                        'constructorId': 'mclaren',
                    }
                ]
            ),
            race_constructor_standings=pd.DataFrame(
                [
                    {'year': 2025, 'round': 24, 'constructorId': 'mclaren', 'points': 833.0},
                ]
            ),
            race_driver_standings=pd.DataFrame(
                [
                    {
                        'year': 2025,
                        'round': 24,
                        'positionDisplayOrder': 1,
                        'driverId': 'lando-norris',
                        'points': 423.0,
                    },
                    {
                        'year': 2025,
                        'round': 24,
                        'positionDisplayOrder': 20,
                        'driverId': 'sergio-perez',
                        'points': 0.0,
                    },
                ]
            ),
            race_results=pd.DataFrame(
                [
                    {
                        'raceId': 1,
                        'year': 2025,
                        'round': 24,
                        'driverId': 'lando-norris',
                        'constructorId': 'mclaren',
                        'positionDisplayOrder': 1,
                        'positionNumber': 1,
                        'points': 25.0,
                    },
                    {
                        'raceId': 2,
                        'year': 2026,
                        'round': 1,
                        'driverId': 'sergio-perez',
                        'constructorId': 'cadillac',
                        'positionDisplayOrder': 1,
                        'positionNumber': 1,
                        'points': 25.0,
                    },
                ]
            ),
            races=pd.DataFrame(
                [
                    {
                        'id': 1,
                        'year': 2025,
                        'round': 24,
                        'date': pd.Timestamp('2025-12-07'),
                        'officialName': '2025 Test Grand Prix',
                        'circuitId': 'melbourne',
                    },
                    {
                        'id': 2,
                        'year': 2026,
                        'round': 1,
                        'date': pd.Timestamp('2026-03-08'),
                        'officialName': '2026 Test Grand Prix',
                        'circuitId': 'melbourne',
                    },
                ]
            ),
            starting_grid=pd.DataFrame(),
            season_entrants_drivers=pd.DataFrame(
                [
                    {
                        'year': 2026,
                        'driverId': 'lando-norris',
                        'constructorId': 'mclaren',
                        'testDriver': False,
                        'rounds': '1;2',
                    },
                    {
                        'year': 2026,
                        'driverId': 'sergio-perez',
                        'constructorId': 'cadillac',
                        'testDriver': False,
                        'rounds': None,
                    },
                    {
                        'year': 2026,
                        'driverId': 'driver-a',
                        'constructorId': 'mclaren',
                        'testDriver': True,
                        'rounds': '1',
                    },
                ]
            ),
        )
        self.f1db_data_to_ml_schema = FakeF1DbDataToMlSchema()
        self.service = F1DbSeasonPrediction(
            loader=FakeLoader(self.raw_data),
            history_selector=None,
            feature_engineer=None,
            pre_qualifying_model=FakeModel([]),
            post_qualifying_model=FakeModel([1.0]),
            f1db_data_to_ml_schema=self.f1db_data_to_ml_schema,
            categorical_categories={'constructorId': [1], 'circuitId': [1]},
        )

    def test_preseason_mode_removes_target_year_actual_results(self):
        simulated_results = self.service._initial_simulated_race_results(
            season_year=2026,
            raw_data=self.raw_data,
            use_best_available_model=False,
        )

        self.assertNotIn(2026, set(simulated_results['year']))

    def test_current_results_mode_keeps_target_year_actual_results(self):
        simulated_results = self.service._initial_simulated_race_results(
            season_year=2026,
            raw_data=self.raw_data,
            use_best_available_model=True,
        )

        self.assertIn(2026, set(simulated_results['year']))

    def test_preseason_calibration_boosts_previous_champion_and_penalizes_inactive_unknown_constructor(self):
        race_predictions = pd.DataFrame(
            [
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 0.0,
                    'predicted_position': 1,
                },
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'sergio-perez',
                    'constructor_ref': 'cadillac',
                    'predicted_score': 0.5,
                    'predicted_position': 2,
                },
            ]
        )

        calibrated = self.service._apply_preseason_calibration(
            season_year=2026,
            raw_data=self.raw_data,
            race_predictions=race_predictions,
        )

        norris_score = calibrated.loc[calibrated['driver_ref'] == 'lando-norris', 'predicted_score'].iloc[0]
        perez_score = calibrated.loc[calibrated['driver_ref'] == 'sergio-perez', 'predicted_score'].iloc[0]

        self.assertGreater(norris_score, perez_score)
        self.assertAlmostEqual(
            perez_score,
            0.5 - INACTIVE_WITH_UNKNOWN_CONSTRUCTOR_SCORE_PENALTY,
        )

    def test_post_qualifying_prediction_passes_date_value_to_schema_builder(self):
        race = self.raw_data.races[self.raw_data.races['id'] == 2].iloc[0]

        self.service._predict_post_qualifying_race(race, self.raw_data)

        self.assertEqual(self.f1db_data_to_ml_schema.received_race_date, date(2026, 3, 8))

    def test_initial_simulated_qualifying_filters_target_year_in_preseason_mode(self):
        preseason_qualifying = self.service._initial_simulated_qualifying(
            season_year=2026,
            raw_data=self.raw_data,
            use_best_available_model=False,
        )
        current_qualifying = self.service._initial_simulated_qualifying(
            season_year=2026,
            raw_data=self.raw_data,
            use_best_available_model=True,
        )

        self.assertTrue(preseason_qualifying.empty)
        self.assertEqual(len(current_qualifying), 1)

    def test_prediction_results_to_race_results_maps_f1_points(self):
        race = self.raw_data.races[self.raw_data.races['id'] == 2].iloc[0]
        race_predictions = pd.DataFrame(
            [
                {'driver_ref': 'lando-norris', 'constructor_ref': 'mclaren', 'predicted_position': 1},
                {'driver_ref': 'sergio-perez', 'constructor_ref': 'cadillac', 'predicted_position': 11},
            ]
        )

        results = self.service._prediction_results_to_race_results(race, race_predictions)

        self.assertEqual(list(results.columns), ['raceId', 'year', 'round', 'driverId', 'constructorId',
                                                 'positionDisplayOrder', 'positionNumber', 'points'])
        self.assertEqual(list(results['points']), [25.0, 0.0])

    def test_apply_known_categories_keeps_known_values_and_nulls_unknown_values(self):
        model_features = pd.DataFrame({'constructorId': [1, 99, None], 'circuitId': [1, 2, 99]})

        result = self.service._apply_known_categories(model_features, ['constructorId', 'circuitId'])

        self.assertEqual(result['constructorId'].iloc[0], 1)
        self.assertTrue(pd.isna(result['constructorId'].iloc[1]))
        self.assertTrue(pd.isna(result['constructorId'].iloc[2]))
        self.assertEqual(result['circuitId'].iloc[0], 1)
        self.assertTrue(pd.isna(result['circuitId'].iloc[1]))
        self.assertTrue(pd.isna(result['circuitId'].iloc[2]))

    def test_preseason_roster_penalty_handles_active_inactive_and_unknown_constructor(self):
        race_counts = pd.Series({'lando-norris': 24})

        self.assertEqual(
            self.service._preseason_roster_penalty('lando-norris', 'mclaren', race_counts, {'mclaren'}),
            0.0,
        )
        self.assertEqual(
            self.service._preseason_roster_penalty('rookie', 'mclaren', race_counts, {'mclaren'}),
            season_prediction.INACTIVE_PREVIOUS_SEASON_SCORE_PENALTY,
        )
        self.assertEqual(
            self.service._preseason_roster_penalty('rookie', 'cadillac', race_counts, {'mclaren'}),
            season_prediction.INACTIVE_WITH_UNKNOWN_CONSTRUCTOR_SCORE_PENALTY,
        )

    def test_previous_season_strength_returns_empty_frames_when_no_previous_standings_exist(self):
        self.assertTrue(self.service._previous_season_driver_strength(2024, self.raw_data).empty)
        self.assertTrue(self.service._previous_season_constructor_strength(2024, self.raw_data).empty)

    def test_previous_season_boost_returns_zero_for_unknown_driver_and_constructor(self):
        self.assertEqual(
            self.service._previous_season_boost(
                driver_ref='unknown-driver',
                constructor_ref='unknown-constructor',
                previous_driver_strength=pd.DataFrame(columns=['driver_ref', 'driver_score_boost']),
                previous_constructor_strength=pd.DataFrame(columns=['constructor_ref', 'constructor_score_boost']),
            ),
            0.0,
        )

    def test_load_training_categories_uses_train_and_test_values(self):
        with patch.object(season_prediction.pd, 'read_csv') as read_csv:
            read_csv.side_effect = [
                pd.DataFrame({'constructorId': [1, 2], 'circuitId': [10, None]}),
                pd.DataFrame({'constructorId': [3, None], 'circuitId': [11, 12]}),
            ]

            categories = F1DbSeasonPrediction._load_training_categories(Path('data'))

        self.assertEqual(categories['constructorId'], [1, 2, 3])
        self.assertEqual(categories['circuitId'], [10, 11, 12])

    def test_create_default_wires_default_dependencies_without_loading_real_models(self):
        with (
            patch.object(season_prediction, 'XGBRanker') as ranker_class,
            patch.object(season_prediction, 'F1DbDataLoader') as loader_class,
            patch.object(season_prediction, 'RaceHistorySelector') as history_selector_class,
            patch.object(season_prediction, 'RaceFeatureEngineering') as feature_engineering_class,
            patch.object(season_prediction.F1DbDataToMlSchema, 'create_default') as schema_create_default,
            patch.object(F1DbSeasonPrediction, '_load_training_categories', return_value={'constructorId': [1]})
        ):
            ranker_class.side_effect = [MagicMock(name='pre_qualifying_model'), MagicMock(name='post_qualifying_model')]

            service = F1DbSeasonPrediction.create_default()

        self.assertIsInstance(service, F1DbSeasonPrediction)
        self.assertEqual(ranker_class.call_count, 2)
        self.assertEqual(loader_class.call_count, 1)
        self.assertEqual(history_selector_class.call_count, 1)
        self.assertEqual(feature_engineering_class.call_count, 1)
        self.assertEqual(schema_create_default.call_count, 1)

    def test_module_helpers_build_entries_standings_and_form_adjusted_points(self):
        race = self.raw_data.races[self.raw_data.races['id'] == 2].iloc[0]

        self.assertTrue(season_prediction._driver_entered_round('1;3', 1))
        self.assertFalse(season_prediction._driver_entered_round(None, 1))

        entries = season_prediction._entries_for_race(self.raw_data.season_entrants_drivers, race)
        self.assertEqual(entries.to_dict('records'), [{'driverId': 'lando-norris', 'constructorId': 'mclaren'}])

        fallback_entries = self.raw_data.season_entrants_drivers.copy()
        fallback_entries['rounds'] = '2'
        fallback = season_prediction._entries_for_race(fallback_entries, race)
        self.assertEqual(
            fallback.to_dict('records'),
            [
                {'driverId': 'lando-norris', 'constructorId': 'mclaren'},
                {'driverId': 'sergio-perez', 'constructorId': 'cadillac'},
            ],
        )

        future_predictions = pd.DataFrame(
            [
                {
                    'race_id': 3,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'points': 25.0,
                },
                {
                    'race_id': 3,
                    'driver_ref': 'sergio-perez',
                    'constructor_ref': 'cadillac',
                    'points': 18.0,
                },
            ]
        )
        current_drivers = pd.DataFrame(
            [{'driver_ref': 'lando-norris', 'constructor_ref': 'mclaren', 'current_points': 50.0}]
        )
        current_constructors = pd.DataFrame([{'constructor_ref': 'mclaren', 'current_points': 80.0}])

        driver_standings = season_prediction._combine_driver_standings(
            current_drivers,
            future_predictions,
            self.raw_data,
            completed_races_count=2,
            current_form_weight=0.5,
        )
        constructor_standings = season_prediction._combine_constructor_standings(
            current_constructors,
            future_predictions,
            self.raw_data,
            completed_races_count=2,
            current_form_weight=0.5,
        )

        self.assertEqual(driver_standings.iloc[0]['driver_ref'], 'lando-norris')
        self.assertEqual(driver_standings.iloc[0]['season_points'], 75.0)
        self.assertEqual(constructor_standings.iloc[0]['constructor_ref'], 'mclaren')
        self.assertEqual(constructor_standings.iloc[0]['constructor_points'], 112.5)

    def test_future_points_helpers_return_empty_shapes_for_empty_predictions(self):
        empty_predictions = pd.DataFrame()

        self.assertEqual(
            list(season_prediction._future_driver_points(empty_predictions).columns),
            ['driver_ref', 'future_constructor_ref', 'future_points'],
        )
        self.assertEqual(
            list(season_prediction._future_constructor_points(empty_predictions).columns),
            ['constructor_ref', 'future_points'],
        )
        self.assertEqual(season_prediction._future_races_count(empty_predictions), 0)

    def test_latest_standings_and_constructor_lookup_support_empty_and_entries_fallback(self):
        self.assertTrue(season_prediction._latest_driver_standings(2024, self.raw_data).empty)
        self.assertTrue(season_prediction._latest_constructor_standings(2024, self.raw_data).empty)

        lookup_from_results = season_prediction._driver_constructor_lookup(2026, self.raw_data)
        self.assertEqual(lookup_from_results['sergio-perez'], 'cadillac')

        raw_without_current_results = self.raw_data
        raw_without_current_results = raw_without_current_results.__class__(
            constructors=raw_without_current_results.constructors,
            drivers=raw_without_current_results.drivers,
            qualifying=raw_without_current_results.qualifying,
            race_constructor_standings=raw_without_current_results.race_constructor_standings,
            race_driver_standings=raw_without_current_results.race_driver_standings,
            race_results=raw_without_current_results.race_results[
                raw_without_current_results.race_results['year'] != 2026
                ],
            races=raw_without_current_results.races,
            starting_grid=raw_without_current_results.starting_grid,
            season_entrants_drivers=raw_without_current_results.season_entrants_drivers,
        )

        lookup_from_entries = season_prediction._driver_constructor_lookup(2026, raw_without_current_results)
        self.assertEqual(lookup_from_entries['lando-norris'], 'mclaren')

    def test_predict_season_standings_handles_empty_preseason_and_current_results_modes(self):
        with patch.object(self.service, '_predict_season_races', return_value=pd.DataFrame()):
            driver_standings, constructor_standings = self.service.predict_season_standings(2026)

        self.assertTrue(driver_standings.empty)
        self.assertTrue(constructor_standings.empty)

        with patch.object(
                self.service,
                '_predict_season_standings_from_current_results',
                return_value=(pd.DataFrame({'a': [1]}), pd.DataFrame({'b': [2]})),
        ) as current_results:
            driver_standings, constructor_standings = self.service.predict_season_standings(
                2026,
                use_current_results=True,
                current_form_weight=0.25,
            )

        current_results.assert_called_once()
        self.assertEqual(list(driver_standings['a']), [1])
        self.assertEqual(list(constructor_standings['b']), [2])

    def test_predict_season_standings_builds_preseason_tables_with_mapped_points(self):
        race_predictions = pd.DataFrame(
            [
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 1.0,
                    'predicted_position': 1,
                },
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'sergio-perez',
                    'constructor_ref': 'cadillac',
                    'predicted_score': 0.5,
                    'predicted_position': 11,
                },
            ]
        )

        with patch.object(self.service, '_predict_season_races', return_value=race_predictions):
            driver_standings, constructor_standings = self.service.predict_season_standings(2026)

        self.assertEqual(driver_standings.iloc[0]['season_points'], 25.0)
        self.assertEqual(constructor_standings.iloc[0]['constructor_points'], 25.0)

    def test_predict_current_results_standings_combines_current_points_and_future_predictions(self):
        future_predictions = pd.DataFrame(
            [
                {
                    'race_id': 3,
                    'race_name': 'Future Grand Prix',
                    'round': 2,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 1.0,
                    'predicted_position': 1,
                }
            ]
        )
        raw_data = self.raw_data.__class__(
            constructors=self.raw_data.constructors,
            drivers=self.raw_data.drivers,
            qualifying=self.raw_data.qualifying,
            race_constructor_standings=pd.DataFrame(
                [{'year': 2026, 'round': 1, 'constructorId': 'mclaren', 'points': 40.0}]
            ),
            race_driver_standings=pd.DataFrame(
                [{'year': 2026, 'round': 1, 'driverId': 'lando-norris', 'points': 25.0}]
            ),
            race_results=self.raw_data.race_results,
            races=self.raw_data.races,
            starting_grid=self.raw_data.starting_grid,
            season_entrants_drivers=self.raw_data.season_entrants_drivers,
        )

        with patch.object(self.service, '_predict_season_races', return_value=future_predictions):
            driver_standings, constructor_standings = self.service._predict_season_standings_from_current_results(
                season_year=2026,
                raw_data=raw_data,
                current_form_weight=0.0,
            )

        self.assertEqual(driver_standings.iloc[0]['season_points'], 50.0)
        self.assertEqual(constructor_standings.iloc[0]['constructor_points'], 65.0)

    def test_predict_season_delegates_to_predict_races_with_loaded_data(self):
        with patch.object(self.service, '_predict_season_races',
                          return_value=pd.DataFrame({'race_id': [2]})) as predict:
            result = self.service.predict_season(2026)

        predict.assert_called_once_with(2026, self.raw_data)
        self.assertEqual(list(result['race_id']), [2])

    def test_predict_race_pre_qualifying_builds_features_and_predicts_single_race(self):
        race_predictions = pd.DataFrame(
            [
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 1.0,
                    'predicted_position': 1,
                }
            ]
        )

        with (
            patch.object(self.service, '_build_pre_qualifying_race_features', return_value=pd.DataFrame({'x': [1]}))
            as build_features,
            patch.object(self.service, '_predict_pre_qualifying_race', return_value=race_predictions) as predict,
        ):
            result = self.service.predict_race_pre_qualifying(date(2026, 3, 8))

        build_features.assert_called_once()
        predict.assert_called_once()
        self.assertEqual(list(result['race_id']), [2])

    def test_predict_season_races_handles_empty_preseason_and_best_available_paths(self):
        empty_raw_data = self.raw_data.__class__(
            constructors=self.raw_data.constructors,
            drivers=self.raw_data.drivers,
            qualifying=self.raw_data.qualifying,
            race_constructor_standings=self.raw_data.race_constructor_standings,
            race_driver_standings=self.raw_data.race_driver_standings,
            race_results=self.raw_data.race_results,
            races=self.raw_data.races[self.raw_data.races['year'] == 2025],
            starting_grid=self.raw_data.starting_grid,
            season_entrants_drivers=self.raw_data.season_entrants_drivers,
        )
        self.assertTrue(self.service._predict_season_races(2026, empty_raw_data).empty)

        post_predictions = pd.DataFrame(
            [
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'driver-a',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 1.0,
                    'predicted_position': 1,
                }
            ]
        )
        with patch.object(self.service, '_predict_post_qualifying_race', return_value=post_predictions) as post:
            result = self.service._predict_season_races(
                season_year=2026,
                raw_data=self.raw_data,
                use_best_available_model=True,
            )

        post.assert_not_called()
        self.assertTrue(result.empty)

        future_raw_data = self.raw_data.__class__(
            constructors=self.raw_data.constructors,
            drivers=self.raw_data.drivers,
            qualifying=self.raw_data.qualifying,
            race_constructor_standings=self.raw_data.race_constructor_standings,
            race_driver_standings=self.raw_data.race_driver_standings,
            race_results=self.raw_data.race_results[self.raw_data.race_results['year'] != 2026],
            races=self.raw_data.races,
            starting_grid=self.raw_data.starting_grid,
            season_entrants_drivers=self.raw_data.season_entrants_drivers,
        )
        with patch.object(self.service, '_predict_post_qualifying_race', return_value=post_predictions):
            result = self.service._predict_season_races(
                season_year=2026,
                raw_data=future_raw_data,
                use_best_available_model=True,
            )

        self.assertEqual(list(result['race_id']), [2])

    def test_predict_season_races_uses_prequalifying_path_and_appends_simulated_results(self):
        race_predictions = pd.DataFrame(
            [
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 1.0,
                    'predicted_position': 1,
                }
            ]
        )
        future_raw_data = self.raw_data.__class__(
            constructors=self.raw_data.constructors,
            drivers=self.raw_data.drivers,
            qualifying=pd.DataFrame(columns=self.raw_data.qualifying.columns),
            race_constructor_standings=self.raw_data.race_constructor_standings,
            race_driver_standings=self.raw_data.race_driver_standings,
            race_results=self.raw_data.race_results[self.raw_data.race_results['year'] != 2026],
            races=self.raw_data.races,
            starting_grid=self.raw_data.starting_grid,
            season_entrants_drivers=self.raw_data.season_entrants_drivers,
        )

        with (
            patch.object(self.service, '_build_pre_qualifying_race_features', return_value=pd.DataFrame({'x': [1]})),
            patch.object(self.service, '_predict_pre_qualifying_race', return_value=race_predictions),
        ):
            result = self.service._predict_season_races(
                season_year=2026,
                raw_data=future_raw_data,
                use_best_available_model=True,
            )

        self.assertEqual(list(result['race_id']), [2])

    def test_predict_season_races_applies_preseason_calibration_when_not_using_best_available_model(self):
        race_predictions = pd.DataFrame(
            [
                {
                    'race_id': 2,
                    'race_name': '2026 Test Grand Prix',
                    'round': 1,
                    'driver_ref': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    'predicted_score': 1.0,
                    'predicted_position': 1,
                }
            ]
        )

        with (
            patch.object(self.service, '_build_pre_qualifying_race_features', return_value=pd.DataFrame({'x': [1]})),
            patch.object(self.service, '_predict_pre_qualifying_race', return_value=race_predictions),
            patch.object(self.service, '_apply_preseason_calibration', return_value=race_predictions) as calibration,
        ):
            result = self.service._predict_season_races(
                season_year=2026,
                raw_data=self.raw_data,
                use_best_available_model=False,
            )

        calibration.assert_called_once()
        self.assertEqual(list(result['race_id']), [2])

    def test_build_pre_qualifying_race_features_uses_history_and_feature_engineering(self):
        race = self.raw_data.races[self.raw_data.races['id'] == 2].iloc[0]
        race_results = pd.DataFrame(
            [
                {
                    'raceId': 1,
                    'year': 2025,
                    'round': 24,
                    'driverId': 'lando-norris',
                    'constructorId': 'mclaren',
                    'positionDisplayOrder': 1,
                    'positionNumber': 1,
                    'points': 25.0,
                }
            ]
        )
        context = RaceContext(
            qualifying=pd.DataFrame(),
            race_id_to_predict=2,
            race_results=race_results,
            race_to_predict=race,
            races_history=self.raw_data.races,
            round_to_predict=1,
        )
        lookups = RaceFeatureLookups(
            circuit_dnf_rate=0.1,
            constructor_dnf_lookup=pd.DataFrame([{'constructorId': 'mclaren', 'constructor_dnf_rate': 0.0}]),
            constructor_momentum_lookup=pd.DataFrame([{'constructorId': 'mclaren', 'constructor_momentum': 1.0}]),
            constructor_track_lookup=pd.DataFrame([{'constructorId': 'mclaren', 'constructor_track_affinity': 1.0}]),
            driver_dnf_lookup=pd.DataFrame([{'driverId': 'lando-norris', 'driver_dnf_rate': 0.0}]),
            driver_momentum_lookup=pd.DataFrame([{'driverId': 'lando-norris', 'driver_momentum': 1.0}]),
            driver_track_lookup=pd.DataFrame([{'driverId': 'lando-norris', 'driver_track_affinity': 1.0}]),
        )
        service = F1DbSeasonPrediction(
            loader=FakeLoader(self.raw_data),
            history_selector=FakeHistorySelector(context),
            feature_engineer=FakeFeatureEngineer(lookups),
            pre_qualifying_model=FakeModel([]),
            post_qualifying_model=FakeModel([]),
            f1db_data_to_ml_schema=self.f1db_data_to_ml_schema,
            categorical_categories={'constructorId': [1], 'circuitId': [1]},
        )

        with patch.object(season_prediction.f1db_utils, 'load_mappings', return_value={
            'constructors': {'mclaren': 1},
            'circuits': {'melbourne': 1},
        }):
            features = service._build_pre_qualifying_race_features(
                raw_data=self.raw_data,
                race_date=date(2026, 3, 8),
                race_results=race_results,
                qualifying=pd.DataFrame(columns=self.raw_data.qualifying.columns),
            )

        self.assertEqual(features.iloc[0]['driverId'], 'lando-norris')
        self.assertEqual(features.iloc[0]['constructorId'], 1)

    def test_predict_pre_qualifying_race_uses_feature_order_and_categories(self):
        race = self.raw_data.races[self.raw_data.races['id'] == 2].iloc[0]
        race_features = pd.DataFrame(
            [
                {
                    'driverId': 'lando-norris',
                    'constructor_ref': 'mclaren',
                    **{column: 0 for column in PRE_QUALIFYING_FEATURE_ORDER},
                }
            ]
        )
        race_features['constructorId'] = 1
        race_features['circuitId'] = 1
        self.service.pre_qualifying_model = FakeModel([0.8])

        result = self.service._predict_pre_qualifying_race(race, race_features)

        self.assertEqual(result.iloc[0]['driver_ref'], 'lando-norris')
        self.assertEqual(list(self.service.pre_qualifying_model.received_features.columns),
                         PRE_QUALIFYING_FEATURE_ORDER)
