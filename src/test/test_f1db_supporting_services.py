from datetime import date
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from application.services.f1db_data_to_ml_schema import F1DbDataToMlSchema
from data.f1db_loader import F1DbDataLoader, F1DbRawData
from data.race_history_selector import RaceContext, RaceHistorySelector
from data.target_race_data_builder import TargetRaceDataBuilder
from features.current_season_feature_engineering import (
    build_f1db_current_season_features,
    build_processed_current_season_features,
)
from features.race_feature_engineering import RaceFeatureEngineering, RaceFeatureLookups
from mappers.race_prediction_mapper import RacePredictionMapper
from utils import f1db_utils
from test import test_api


def _raw_data() -> F1DbRawData:
    return F1DbRawData(
        constructors=pd.DataFrame([{'id': 'mclaren', 'name': 'McLaren'}]),
        drivers=pd.DataFrame(
            [
                {
                    'id': 'lando-norris',
                    'name': 'Lando Norris',
                    'dateOfBirth': pd.Timestamp('1999-11-13'),
                    'driver_age': 26.0,
                },
                {
                    'id': 'oscar-piastri',
                    'name': 'Oscar Piastri',
                    'dateOfBirth': pd.Timestamp('2001-04-06'),
                    'driver_age': 24.0,
                },
            ]
        ),
        qualifying=pd.DataFrame(
            [
                {
                    'raceId': 2,
                    'year': 2026,
                    'round': 2,
                    'driverId': 'lando-norris',
                    'constructorId': 'mclaren',
                    'q1Millis': 90000.0,
                    'q2Millis': None,
                    'q3Millis': None,
                    'positionDisplayOrder': None,
                }
            ]
        ),
        race_constructor_standings=pd.DataFrame(),
        race_driver_standings=pd.DataFrame(),
        race_results=pd.DataFrame(
            [
                {
                    'raceId': 1,
                    'year': 2026,
                    'round': 1,
                    'driverId': 'lando-norris',
                    'constructorId': 'mclaren',
                    'positionDisplayOrder': 2,
                    'positionNumber': 2,
                    'points': 18.0,
                },
                {
                    'raceId': 1,
                    'year': 2026,
                    'round': 1,
                    'driverId': 'oscar-piastri',
                    'constructorId': 'mclaren',
                    'positionDisplayOrder': 1,
                    'positionNumber': None,
                    'points': 25.0,
                },
            ]
        ),
        races=pd.DataFrame(
            [
                {
                    'id': 1,
                    'year': 2026,
                    'round': 1,
                    'date': pd.Timestamp('2026-03-01'),
                    'officialName': 'Round 1',
                    'circuitId': 'melbourne',
                },
                {
                    'id': 2,
                    'year': 2026,
                    'round': 2,
                    'date': pd.Timestamp('2026-03-08'),
                    'officialName': 'Round 2',
                    'circuitId': 'melbourne',
                },
            ]
        ),
        starting_grid=pd.DataFrame(
            [{'raceId': 2, 'driverId': 'lando-norris', 'positionDisplayOrder': None}]
        ),
        season_entrants_drivers=pd.DataFrame(),
    )


def _lookups() -> RaceFeatureLookups:
    return RaceFeatureLookups(
        circuit_dnf_rate=0.5,
        constructor_dnf_lookup=pd.DataFrame([{'constructorId': 'mclaren', 'constructor_dnf_rate': 0.5}]),
        constructor_momentum_lookup=pd.DataFrame([{'constructorId': 'mclaren', 'constructor_momentum': 1.5}]),
        constructor_track_lookup=pd.DataFrame([{'constructorId': 'mclaren', 'constructor_track_affinity': 1.5}]),
        driver_dnf_lookup=pd.DataFrame([{'driverId': 'lando-norris', 'driver_dnf_rate': 0.0}]),
        driver_momentum_lookup=pd.DataFrame([{'driverId': 'lando-norris', 'driver_momentum': 2.0}]),
        driver_track_lookup=pd.DataFrame([{'driverId': 'lando-norris', 'driver_track_affinity': 2.0}]),
    )


class F1DbSupportingServicesTest(TestCase):
    def test_loader_reads_all_raw_data_and_converts_dates(self):
        def read_csv(path, low_memory):
            file_name = Path(path).name
            self.assertFalse(low_memory)
            if file_name == 'f1db-races.csv':
                return pd.DataFrame([{'date': '2026-03-08'}])
            if file_name == 'f1db-drivers.csv':
                return pd.DataFrame([{'dateOfBirth': '1999-11-13'}])
            return pd.DataFrame([{'source': file_name}])

        with patch('data.f1db_loader.pd.read_csv', side_effect=read_csv):
            raw_data = F1DbDataLoader(Path('project')).load()

        self.assertEqual(raw_data.races['date'].iloc[0], pd.Timestamp('2026-03-08'))
        self.assertEqual(raw_data.drivers['dateOfBirth'].iloc[0], pd.Timestamp('1999-11-13'))
        self.assertEqual(raw_data.constructors['source'].iloc[0], 'f1db-constructors.csv')
        self.assertEqual(raw_data.season_entrants_drivers['source'].iloc[0], 'f1db-seasons-entrants-drivers.csv')

    def test_loader_read_csv_uses_raw_data_path(self):
        with patch('data.f1db_loader.pd.read_csv', return_value=pd.DataFrame([{'ok': True}])) as read_csv:
            result = F1DbDataLoader(Path('project'))._read_csv('file.csv')

        self.assertTrue(result['ok'].iloc[0])
        read_csv.assert_called_once_with(Path('project') / 'validation_data' / '01_raw' / 'file.csv',
                                         low_memory=False)

    def test_race_history_selector_filters_history_to_requested_date(self):
        raw_data = _raw_data()

        context = RaceHistorySelector().select(raw_data, date(2026, 3, 8))

        self.assertEqual(context.race_id_to_predict, 2)
        self.assertEqual(context.round_to_predict, 2)
        self.assertEqual(list(context.races_history['id']), [1, 2])
        self.assertEqual(list(context.race_results['raceId']), [1, 1])
        self.assertEqual(list(context.qualifying['raceId']), [2])

    def test_target_race_data_builder_prepares_and_builds_target_features(self):
        raw_data = _raw_data()
        builder = TargetRaceDataBuilder()
        qualifying = raw_data.qualifying.copy()
        builder.prepare_qualifying_data(qualifying)
        context = RaceContext(
            qualifying=qualifying,
            race_id_to_predict=2,
            race_results=raw_data.race_results,
            race_to_predict=raw_data.races[raw_data.races['id'] == 2].iloc[0],
            races_history=raw_data.races,
            round_to_predict=2,
        )

        with patch.object(f1db_utils, 'load_mappings', return_value={
            'constructors': {'mclaren': 1},
            'circuits': {'melbourne': 10},
        }):
            target = builder.build(raw_data, context, _lookups())

        self.assertTrue(target['reached_q1'].iloc[0])
        self.assertFalse(target['reached_q2'].iloc[0])
        self.assertEqual(target['q2_millis'].iloc[0], 0.0)
        self.assertIsNone(target['position_qualifying'].iloc[0])
        self.assertIsNone(target['grid'].iloc[0])
        self.assertEqual(target['constructor_id_mapped'].iloc[0], 1)
        self.assertEqual(target['circuit_id_mapped'].iloc[0], 10)
        self.assertEqual(target['circuit_dnf_rate'].iloc[0], 0.5)

    def test_target_race_data_builder_raises_when_qualifying_is_missing(self):
        raw_data = _raw_data()
        context = RaceContext(
            qualifying=pd.DataFrame(columns=raw_data.qualifying.columns),
            race_id_to_predict=2,
            race_results=raw_data.race_results,
            race_to_predict=raw_data.races[raw_data.races['id'] == 2].iloc[0],
            races_history=raw_data.races,
            round_to_predict=2,
        )

        TargetRaceDataBuilder().build(raw_data, context, _lookups())

    def test_race_feature_engineering_builds_all_lookup_tables(self):
        raw_data = _raw_data()
        race_to_predict = raw_data.races[raw_data.races['id'] == 2].iloc[0]
        race_results_past = pd.merge(
            raw_data.race_results,
            raw_data.races[['id', 'date', 'circuitId']],
            left_on='raceId',
            right_on='id',
            how='inner',
        )

        lookups = RaceFeatureEngineering().build_features(race_results_past, race_to_predict)

        self.assertEqual(lookups.circuit_dnf_rate, 0.5)
        self.assertEqual(lookups.driver_momentum_lookup.loc[0, 'driver_momentum'], 2.0)
        self.assertEqual(lookups.constructor_dnf_lookup.loc[0, 'constructor_dnf_rate'], 0.5)
        self.assertEqual(lookups.constructor_momentum_lookup.loc[0, 'constructor_momentum'], 1.5)
        self.assertEqual(lookups.driver_track_lookup.loc[0, 'driver_track_affinity'], 2.0)

    def test_current_season_features_use_prior_processed_rows_and_fallback_columns(self):
        features = pd.DataFrame(
            [
                {'constructorId': 1},
                {'constructorId': 1},
                {'constructorId': 1},
                {'constructorId': 1},
            ]
        )

        result = build_processed_current_season_features(
            features=features,
            y_position=pd.Series([1, 4, 2, 8]),
            qid=pd.Series([1, 2, 3, 4]),
            year=pd.Series([2026, 2026, 2026, 2026]),
            driver_id=pd.Series(['driver-a', 'driver-a', 'driver-a', 'driver-a']),
        )

        self.assertEqual(result.loc[0, 'current_season_points_per_race'], 0.0)
        self.assertEqual(result.loc[3, 'current_season_points_per_race'], 18.333333333333332)
        self.assertEqual(result.loc[3, 'current_season_avg_finish'], 2.3333333333333335)
        self.assertEqual(result.loc[3, 'current_season_podium_rate'], 2 / 3)
        self.assertEqual(result.loc[3, 'last_3_current_season_avg_finish'], 2.3333333333333335)

    def test_f1db_current_season_features_use_prior_results_and_q3_rate(self):
        race_features = pd.DataFrame([{'driverId': 'lando-norris', 'constructor_ref': 'mclaren'}])
        race_results = pd.DataFrame(
            [
                {
                    'raceId': 1,
                    'year': 2026,
                    'round': 1,
                    'driverId': 'lando-norris',
                    'constructorId': 'mclaren',
                    'positionDisplayOrder': 2,
                    'points': 18.0,
                }
            ]
        )
        qualifying = pd.DataFrame(
            [{'raceId': 1, 'year': 2026, 'round': 1, 'driverId': 'lando-norris', 'q3Millis': 88000.0}]
        )
        race = pd.Series({'year': 2026, 'round': 2})

        result = build_f1db_current_season_features(race_features, race_results, qualifying, race)

        self.assertEqual(result.loc[0, 'current_season_points_per_race'], 18.0)
        self.assertEqual(result.loc[0, 'current_season_avg_finish'], 2.0)
        self.assertEqual(result.loc[0, 'current_season_podium_rate'], 1.0)
        self.assertEqual(result.loc[0, 'current_season_q3_rate'], 1.0)

    def test_f1db_current_season_features_return_zeroes_without_prior_rows(self):
        result = build_f1db_current_season_features(
            race_features=pd.DataFrame([{'driverId': 'rookie', 'constructor_ref': 'mclaren'}]),
            race_results=pd.DataFrame(columns=['year', 'round', 'raceId', 'driverId', 'constructorId', 'points']),
            qualifying=pd.DataFrame(columns=['year', 'round', 'driverId', 'q3Millis']),
            race=pd.Series({'year': 2026, 'round': 1}),
        )

        self.assertEqual(result.loc[0, 'current_season_points_per_race'], 0.0)
        self.assertEqual(result.loc[0, 'current_season_avg_finish'], 0.0)
        self.assertEqual(result.loc[0, 'current_season_q3_rate'], 0.0)

    def test_mapper_builds_request_with_defaults_for_missing_grid_and_categories(self):
        target = pd.DataFrame(
            [
                {
                    'driverId': 'lando-norris',
                    'driver_age': 26.0,
                    'driver_momentum': 2.0,
                    'current_season_points_per_race': 18.0,
                    'current_season_avg_finish': 2.0,
                    'current_season_podium_rate': 1.0,
                    'current_season_q3_rate': 0.0,
                    'driver_track_affinity': 2.0,
                    'driver_dnf_rate': 0.0,
                    'position_qualifying': None,
                    'grid': None,
                    'q1_millis': 90000.0,
                    'reached_q1': True,
                    'q2_millis': 0.0,
                    'reached_q2': False,
                    'q3_millis': 0.0,
                    'reached_q3': False,
                    'constructor_id_mapped': None,
                    'constructor_momentum': 1.5,
                    'current_constructor_points_per_race': 43.0,
                    'constructor_track_affinity': 1.5,
                    'constructor_dnf_rate': 0.5,
                    'circuit_id_mapped': None,
                    'last_3_current_season_avg_finish': 2.0,
                    'circuit_dnf_rate': 0.5,
                }
            ]
        )

        request = RacePredictionMapper().to_request(target, 'Australian Grand Prix', 2)

        self.assertEqual(request.race_name, 'Australian Grand Prix')
        self.assertEqual(len(request.grid_data), 1)
        self.assertEqual(request.grid_data[0].position_qualifying, 20)
        self.assertEqual(request.grid_data[0].grid, 20)
        self.assertIsNone(request.grid_data[0].constructor_id)
        self.assertIsNone(request.grid_data[0].circuit_id)

    def test_schema_service_default_factory_and_build_request(self):
        raw_data = _raw_data()
        context = RaceContext(
            qualifying=raw_data.qualifying.copy(),
            race_id_to_predict=2,
            race_results=raw_data.race_results,
            race_to_predict=raw_data.races[raw_data.races['id'] == 2].iloc[0],
            races_history=raw_data.races,
            round_to_predict=2,
        )
        loader = SimpleNamespace(load=MagicMock(return_value=raw_data))
        history_selector = SimpleNamespace(select=MagicMock(return_value=context))
        feature_engineer = SimpleNamespace(build_features=MagicMock(return_value=_lookups()))
        builder = TargetRaceDataBuilder()
        mapper = RacePredictionMapper()
        service = F1DbDataToMlSchema(loader, history_selector, feature_engineer, builder, mapper)

        with patch.object(f1db_utils, 'load_mappings', return_value={
            'constructors': {'mclaren': 1},
            'circuits': {'melbourne': 10},
        }):
            request = service.build_request(date(2026, 3, 8))

        self.assertEqual(request.race_name, 'Round 2')
        self.assertEqual(request.grid_data[0].driver_ref, 'lando-norris')
        self.assertEqual(request.grid_data[0].constructor_id, 1)
        self.assertEqual(request.grid_data[0].circuit_id, 10)

        with (
            patch('application.services.f1db_data_to_ml_schema.F1DbDataLoader') as loader_class,
            patch('application.services.f1db_data_to_ml_schema.RaceHistorySelector') as selector_class,
            patch('application.services.f1db_data_to_ml_schema.RaceFeatureEngineering') as engineer_class,
            patch('application.services.f1db_data_to_ml_schema.TargetRaceDataBuilder') as builder_class,
            patch('application.services.f1db_data_to_ml_schema.RacePredictionMapper') as mapper_class,
        ):
            default_service = F1DbDataToMlSchema.create_default()

        self.assertIsInstance(default_service, F1DbDataToMlSchema)
        self.assertEqual(loader_class.call_count, 1)
        self.assertEqual(selector_class.call_count, 1)
        self.assertEqual(engineer_class.call_count, 1)
        self.assertEqual(builder_class.call_count, 1)
        self.assertEqual(mapper_class.call_count, 1)

    def test_utils_load_mappings_and_fetch_raw_data(self):
        with patch('builtins.open', mock_open(read_data='{"constructors": {"mclaren": 1}}')):
            mappings = f1db_utils.load_mappings()

        self.assertEqual(mappings, {'constructors': {'mclaren': 1}})

        response = SimpleNamespace(content=b'zip-bytes', raise_for_status=MagicMock())
        zip_file = MagicMock()
        with (
            patch.object(f1db_utils.requests, 'get', return_value=response) as get,
            patch.object(f1db_utils.zipfile, 'ZipFile', return_value=zip_file) as zip_class,
            patch.object(f1db_utils.io, 'BytesIO', return_value='bytes-buffer') as bytes_io,
        ):
            f1db_utils.fetch_and_extract_raw_data(SimpleNamespace(project_root=Path('project')))

        get.assert_called_once()
        response.raise_for_status.assert_called_once()
        bytes_io.assert_called_once_with(b'zip-bytes')
        zip_class.assert_called_once_with('bytes-buffer')
        zip_file.__enter__.return_value.extractall.assert_called_once_with(
            Path('project') / 'validation_data' / '01_raw'
        )

    def test_manual_api_helper_prints_success_and_error_responses_without_calling_real_api(self):
        success = SimpleNamespace(status_code=200, json=MagicMock(return_value={'ok': True}))
        failure = SimpleNamespace(status_code=500, json=MagicMock(return_value={'error': 'bad'}))

        with (
            patch.object(test_api.requests, 'post', side_effect=[success, failure]) as post,
            patch('builtins.print') as print_mock,
        ):
            test_api.setup_api_response('http://localhost/test', {'payload': True})

        self.assertEqual(post.call_count, 2)
        self.assertIn('model_type=regressor', post.call_args_list[0].args[0])
        self.assertIn('model_type=ranker', post.call_args_list[1].args[0])
        self.assertGreater(print_mock.call_count, 0)

    def test_manual_api_helper_prints_regressor_error_and_ranker_success(self):
        failure = SimpleNamespace(status_code=500, json=MagicMock(return_value={'error': 'bad'}))
        success = SimpleNamespace(status_code=200, json=MagicMock(return_value={'ok': True}))

        with (
            patch.object(test_api.requests, 'post', side_effect=[failure, success]) as post,
            patch('builtins.print') as print_mock,
        ):
            test_api.setup_api_response('http://localhost/test', {'payload': True})

        self.assertEqual(post.call_count, 2)
        self.assertGreater(print_mock.call_count, 0)

    def test_manual_prediction_endpoint_helpers_build_expected_requests(self):
        with patch.object(test_api, 'setup_api_response') as setup_api_response:
            test_api.test_prediction_endpoint()
            test_api.test_prediction_by_race_date_endpoint()

        self.assertEqual(setup_api_response.call_count, 2)
        self.assertEqual(setup_api_response.call_args_list[0].args[0], 'http://localhost:8000/api/v1/predict/race')
        self.assertEqual(
            setup_api_response.call_args_list[1].args[0],
            'http://localhost:8000/api/v1/predict/race-by-date',
        )
        self.assertEqual(setup_api_response.call_args_list[1].args[1], {'race_date': '2025-12-07'})

    def test_manual_season_endpoint_helpers_build_expected_requests(self):
        response = SimpleNamespace(json=MagicMock(return_value={'drivers': []}))

        with (
            patch.object(test_api.requests, 'post', return_value=response) as post,
            patch('builtins.print'),
        ):
            test_api.test_season_prediction_endpoint()
            test_api.test_season_prediction_during_season_endpoint()

        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].args[0], 'http://localhost:8000/api/v1/predict/season')
        self.assertEqual(post.call_args_list[0].kwargs['json'], {'year': 2026, 'use_current_results': False})
        self.assertEqual(post.call_args_list[1].kwargs['json'], {'year': 2026, 'use_current_results': True})

    def test_manual_api_module_main_runs_during_season_helper(self):
        response = SimpleNamespace(json=MagicMock(return_value={'drivers': []}))

        with (
            patch('requests.post', return_value=response) as post,
            patch('builtins.print'),
        ):
            runpy.run_path(Path(__file__).parent / 'test_api.py', run_name='__main__')

        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], 'http://localhost:8000/api/v1/predict/season')
        self.assertEqual(post.call_args.kwargs['json'], {'year': 2026, 'use_current_results': True})
