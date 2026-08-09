from typing import Any

import pandas as pd
from pandas import DataFrame

from api.schemas.predict_dto import DriverRaceData, RacePredictionRequest


class RacePredictionMapper:
    def to_request(self, target_race_df: DataFrame, race_name: str, round_to_predict: int) -> RacePredictionRequest:
        return RacePredictionRequest(
            race_name=race_name,
            grid_data=self._build_grid_payload(round_to_predict, target_race_df),
        )

    def _build_grid_payload(self, round_to_predict: int, target_race_df: DataFrame) -> list[Any]:
        grid_payload = []

        for _, data in target_race_df.iterrows():
            driver_obj = {
                'driver_ref': str(data['driverId']),
                'driver_age': float(data['driver_age']),
                'driver_momentum': float(data['driver_momentum']),
                'current_season_points_per_race': float(data['current_season_points_per_race']),
                'current_season_avg_finish': float(data['current_season_avg_finish']),
                'current_season_podium_rate': float(data['current_season_podium_rate']),
                'current_season_q3_rate': float(data['current_season_q3_rate']),
                'driver_track_affinity': float(data['driver_track_affinity']),
                'driver_dnf_rate': float(data['driver_dnf_rate']),

                'position_qualifying': int(data['position_qualifying'])
                if pd.notna(data['position_qualifying'])
                else 20,
                'grid': int(data['grid']) if pd.notna(data['grid']) else 20,
                'q1_millis': float(data['q1_millis']),
                'reached_q1': bool(data['reached_q1']),
                'q2_millis': float(data['q2_millis']),
                'reached_q2': bool(data['reached_q2']),
                'q3_millis': float(data['q3_millis']),
                'reached_q3': bool(data['reached_q3']),

                'constructor_id': int(data['constructor_id_mapped'])
                if pd.notna(data['constructor_id_mapped'])
                else None,
                'constructor_momentum': float(data['constructor_momentum']),
                'current_constructor_points_per_race': float(data['current_constructor_points_per_race']),
                'constructor_track_affinity': float(data['constructor_track_affinity']),
                'constructor_dnf_rate': float(data['constructor_dnf_rate']),

                'round': round_to_predict,
                'circuit_id': int(data['circuit_id_mapped']) if pd.notna(data['circuit_id_mapped']) else None,
                'last_3_current_season_avg_finish': float(data['last_3_current_season_avg_finish']),
                'circuit_dnf_rate': float(data['circuit_dnf_rate']),
            }
            grid_payload.append(DriverRaceData(**driver_obj))

        return grid_payload
