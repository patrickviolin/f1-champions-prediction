from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame


@dataclass(frozen=True)
class F1DbRawData:
    constructors: DataFrame
    drivers: DataFrame
    qualifying: DataFrame
    race_constructor_standings: DataFrame
    race_driver_standings: DataFrame
    race_results: DataFrame
    races: DataFrame
    starting_grid: DataFrame
    season_entrants_drivers: DataFrame


class F1DbDataLoader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.raw_data_path = project_root / 'validation_data' / '01_raw'

    def load(self) -> F1DbRawData:
        races = self._read_csv('f1db-races.csv')
        races['date'] = pd.to_datetime(races['date'])

        drivers = self._read_csv('f1db-drivers.csv')
        drivers['dateOfBirth'] = pd.to_datetime(drivers['dateOfBirth'])

        return F1DbRawData(
            constructors=self._read_csv('f1db-constructors.csv'),
            drivers=drivers,
            qualifying=self._read_csv('f1db-races-qualifying-results.csv'),
            race_constructor_standings=self._read_csv('f1db-races-constructor-standings.csv'),
            race_driver_standings=self._read_csv('f1db-races-driver-standings.csv'),
            race_results=self._read_csv('f1db-races-race-results.csv'),
            races=races,
            starting_grid=self._read_csv('f1db-races-starting-grid-positions.csv'),
            season_entrants_drivers=self._read_csv('f1db-seasons-entrants-drivers.csv'),
        )

    def _read_csv(self, file_name: str) -> DataFrame:
        return pd.read_csv(self.raw_data_path / file_name, low_memory=False)
