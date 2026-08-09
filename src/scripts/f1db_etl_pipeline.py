import json

import requests

from application.services.f1db_data_to_ml_schema import F1DbDataToMlSchema


class F1DbEtlPipeline:
    def __init__(self):
        self.api_url = 'http://localhost:8000/api/v1/predict/race'


if __name__ == '__main__':
    f1db_pipeline = F1DbEtlPipeline()
    # f1db_pipeline.fetch_and_extract_raw_data()

    print("Starting data transformation")
    f1db_data_to_ml_schema = F1DbDataToMlSchema.create_default()
    request_dto = f1db_data_to_ml_schema.build_request(race_date='2026-07-26')

    payload_json = request_dto.model_dump()

    print("Requesting to API")
    response = requests.post(
        f"{f1db_pipeline.api_url}?model_type=ranker",
        json=payload_json,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print("Prediction successful!")
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
