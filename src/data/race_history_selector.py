from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from data.f1db_loader import F1DbRawData


@dataclass(frozen=True)
class RaceContext:
    qualifying: DataFrame
    race_id_to_predict: Any
    race_results: DataFrame
    race_to_predict: Series
    races_history: DataFrame
    round_to_predict: int


class RaceHistorySelector:
    def select(self, raw_data: F1DbRawData, race_date: date) -> RaceContext:
        race_date = pd.to_datetime(race_date)

        races_history = raw_data.races[raw_data.races['date'] <= race_date]
        race_to_predict = races_history[races_history['date'] == race_date].iloc[0]

        race_id_to_predict = race_to_predict['id']
        round_to_predict = int(race_to_predict['round'])

        return RaceContext(
            qualifying=raw_data.qualifying[raw_data.qualifying['raceId'] <= race_id_to_predict].copy(),
            race_id_to_predict=race_id_to_predict,
            race_results=raw_data.race_results[raw_data.race_results['raceId'] <= race_id_to_predict].copy(),
            race_to_predict=race_to_predict,
            races_history=races_history.copy(),
            round_to_predict=round_to_predict,
        )
