import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


class F1DbEtlPipeline:
    def __init__(self):
        self.db_version = 'v2026.11.0'
        self.base_url = f'https://github.com/f1db/f1db/releases/download/{self.db_version}/f1db-csv.zip'
        self.api_url = 'http://localhost:8000/api/v1/predict/race'
        self.project_root = Path(__file__).parent.parent.parent

    def fetch_and_extract_raw_data(self):
        """Download the zip file containing the raw data and extract it in a CSV file saved locally."""
        raw_data = requests.get(self.base_url)
        raw_data.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(raw_data.content)) as zip_ref:
            zip_ref.extractall(self.project_root / 'validation_data' / '01_raw')


if __name__ == '__main__':
    f1db_pipeline = F1DbEtlPipeline()
    f1db_pipeline.fetch_and_extract_raw_data()
