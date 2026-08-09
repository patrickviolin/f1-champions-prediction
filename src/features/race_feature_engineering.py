from dataclasses import dataclass

from pandas import DataFrame, Series


@dataclass(frozen=True)
class RaceFeatureLookups:
    circuit_dnf_rate: float
    constructor_dnf_lookup: DataFrame
    constructor_momentum_lookup: DataFrame
    constructor_track_lookup: DataFrame
    driver_dnf_lookup: DataFrame
    driver_momentum_lookup: DataFrame
    driver_track_lookup: DataFrame


class RaceFeatureEngineering:
    def build_features(self, race_results_past: DataFrame, race_to_predict: Series) -> RaceFeatureLookups:
        race_results_past = race_results_past.copy()
        race_results_past['driver_momentum'] = (
            race_results_past.groupby('driverId')['positionDisplayOrder']
            .transform(lambda series: series.rolling(3, min_periods=1).mean())
        )

        past_in_circuit = race_results_past[race_results_past['circuitId'] == race_to_predict['circuitId']].copy()
        past_in_circuit['driver_track_affinity'] = (
            past_in_circuit.groupby('driverId')['positionDisplayOrder']
            .transform(lambda series: series.expanding(1).mean())
        )

        race_results_past['is_dnf'] = race_results_past['positionNumber'].isna().astype(int)
        race_results_past['driver_dnf_rate'] = (
            race_results_past.groupby('driverId')['is_dnf']
            .transform(lambda series: series.expanding(1).mean())
        )
        race_results_past['constructor_dnf_rate'] = (
            race_results_past.groupby('constructorId')['is_dnf']
            .transform(lambda series: series.expanding(1).mean())
        )

        team_race_totals = (
            race_results_past.groupby(['constructorId', 'raceId', 'date'])['positionDisplayOrder']
            .mean()
            .reset_index()
            .sort_values(by='date')
        )
        team_race_totals['constructor_momentum'] = (
            team_race_totals.groupby('constructorId')['positionDisplayOrder']
            .transform(lambda series: series.rolling(3, min_periods=1).mean())
        )

        team_track_totals = (
            past_in_circuit.groupby(['constructorId', 'raceId', 'date'])['positionDisplayOrder']
            .mean()
            .reset_index()
            .sort_values(by='date')
        )
        team_track_totals['constructor_track_affinity'] = (
            team_track_totals.groupby('constructorId')['positionDisplayOrder']
            .transform(lambda series: series.expanding(1).mean())
        )

        return RaceFeatureLookups(
            circuit_dnf_rate=float(race_results_past['is_dnf'].mean()),
            constructor_dnf_lookup=(
                race_results_past.groupby('constructorId')['constructor_dnf_rate'].last().reset_index()
            ),
            constructor_momentum_lookup=(
                team_race_totals.groupby('constructorId')['constructor_momentum'].last().reset_index()
            ),
            constructor_track_lookup=(
                team_track_totals.groupby('constructorId')['constructor_track_affinity'].last().reset_index()
            ),
            driver_dnf_lookup=race_results_past.groupby('driverId')['driver_dnf_rate'].last().reset_index(),
            driver_momentum_lookup=race_results_past.groupby('driverId')['driver_momentum'].last().reset_index(),
            driver_track_lookup=past_in_circuit.groupby('driverId')['driver_track_affinity'].last().reset_index(),
        )
