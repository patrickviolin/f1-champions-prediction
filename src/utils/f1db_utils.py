import io
import json
import zipfile
from pathlib import Path
from typing import Dict

import requests


def load_mappings() -> Dict[str, int]:
    project_root = Path(__file__).parent.parent.parent

    mapping_path = project_root / 'config' / 'mapping_circuit_and_constructor.json'

    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fetch_and_extract_raw_data(self):
    """Download the zip file containing the raw data and extract it in a CSV file saved locally."""
    db_version = 'v2026.11.0'
    base_url = f'https://github.com/f1db/f1db/releases/download/{db_version}/f1db-csv.zip'

    raw_data = requests.get(base_url)
    raw_data.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(raw_data.content)) as zip_ref:
        zip_ref.extractall(self.project_root / 'validation_data' / '01_raw')
