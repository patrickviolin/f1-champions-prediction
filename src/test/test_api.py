import json

import requests


def setup_api_response(url, payload):
    headers = {
        "Content-Type": "application/json",
    }

    print('=' * 50)
    print('TEST 1: INFERENCE WITH XGBREGRESSOR')
    print('=' * 50)

    response_regressor = requests.post(f"{url}?model_type=regressor", json=payload, headers=headers)

    if response_regressor.status_code == 200:
        print(json.dumps(response_regressor.json(), indent=4, ensure_ascii=False))
    else:
        print(
            f"Error {response_regressor.status_code}: \n{json.dumps(response_regressor.json(), indent=4, ensure_ascii=False)}")

    print('\n' + '=' * 50)
    print('TEST 2: INFERENCE WITH XGBRANKER')
    print('=' * 50)

    response_ranker = requests.post(f"{url}?model_type=ranker", json=payload, headers=headers)

    if response_ranker.status_code == 200:
        print(json.dumps(response_ranker.json(), indent=4, ensure_ascii=False))
    else:
        print(
            f"Error {response_ranker.status_code}: \n{json.dumps(response_ranker.json(), indent=4, ensure_ascii=False)}")


def test_prediction_endpoint():
    url = 'http://localhost:8000/api/v1/predict/race'

    payload = {
        "race_name": "GP de Interlagos 2026",
        "grid_data": [
            {
                "driver_ref": "max_verstappen",
                "driver_age": 28.5,
                "driver_momentum": 15.2,
                "driver_track_affinity": 14.5,
                "driver_dnf_rate": 0.05,
                "position_qualifying": 1,
                "grid": 1,
                "q1_millis": 70100,
                "reached_q1": True,
                "q2_millis": 69500,
                "reached_q2": True,
                "q3_millis": 68900,
                "reached_q3": True,
                "constructor_id": 9,
                "constructor_momentum": 25.0,
                "constructor_track_affinity": 22.0,
                "constructor_dnf_rate": 0.02,
                "round": 20,
                "circuit_id": 18,
                "circuit_dnf_rate": 0.15
            },
            {
                "driver_ref": "lando_norris",
                "driver_age": 26.1,
                "driver_momentum": 14.8,
                "driver_track_affinity": 12.0,
                "driver_dnf_rate": 0.10,
                "position_qualifying": 2,
                "grid": 2,
                "q1_millis": 70250,
                "reached_q1": True,
                "q2_millis": 69650,
                "reached_q2": True,
                "q3_millis": 69100,
                "reached_q3": True,
                "constructor_id": 1,
                "constructor_momentum": 24.5,
                "constructor_track_affinity": 20.0,
                "constructor_dnf_rate": 0.05,
                "round": 20,
                "circuit_id": 18,
                "circuit_dnf_rate": 0.15
            }
        ]
    }

    setup_api_response(url, payload)


def test_prediction_by_race_date_endpoint():
    url = 'http://localhost:8000/api/v1/predict/race-by-date'

    payload = {
        "race_date": '2025-12-07'
    }

    setup_api_response(url, payload)


def test_season_prediction_endpoint():
    url = 'http://localhost:8000/api/v1/predict/season'

    payload = {
        "year": 2026,
        "use_current_results": False
    }

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))


def test_season_prediction_during_season_endpoint():
    url = 'http://localhost:8000/api/v1/predict/season'

    payload = {
        "year": 2026,
        "use_current_results": True
    }

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))


if __name__ == '__main__':
    test_season_prediction_during_season_endpoint()
