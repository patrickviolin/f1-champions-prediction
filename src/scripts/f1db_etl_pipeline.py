import json

import requests

from utils import api_utils


class F1DbEtlPipeline:
    def __init__(self):
        self.api_url = 'http://localhost:8000/api/v1/predict/race'


if __name__ == '__main__':
    f1db_pipeline = F1DbEtlPipeline()
    # f1db_pipeline.fetch_and_extract_raw_data()

    print("Iniciando transformação de dados...")
    request_dto = api_utils.transform_f1db_data_to_api_schema(race_date='2026-07-26')

    payload_json = request_dto.model_dump()

    print("Disparando requisição contra a API...")
    response = requests.post(
        f"{f1db_pipeline.api_url}?model_type=ranker",
        json=payload_json,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print("Predição concluída!")
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    else:
        print(f"Erro na API: {response.status_code}")
        print(response.text)
