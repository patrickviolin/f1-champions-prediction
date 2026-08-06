from datetime import date
from pathlib import Path

import pandas as pd

from api.schemas.predict_dto import DriverRaceData, RacePredictionRequest
from utils import f1db_utils


def transform_f1db_data_to_api_schema(race_date: date) -> RacePredictionRequest:
    """Transform the raw data from F1DB to API schema."""
    project_root = Path(__file__).parent.parent.parent
    mappings = f1db_utils.load_mappings()

    race_date = pd.to_datetime(race_date)

    # Load and Initial Conversion
    races = pd.read_csv(project_root / 'validation_data' / '01_raw' / 'f1db-races.csv')
    races['date'] = pd.to_datetime(races['date'])

    race_results = pd.read_csv(project_root / 'validation_data' / '01_raw' / 'f1db-races-race-results.csv')
    qualifying = pd.read_csv(
        project_root / 'validation_data' / '01_raw' / 'f1db-races-qualifying-results.csv')
    starting_grid = pd.read_csv(
        project_root / 'validation_data' / '01_raw' / 'f1db-races-starting-grid-positions.csv')
    drivers = pd.read_csv(project_root / 'validation_data' / '01_raw' / 'f1db-drivers.csv')

    drivers['dateOfBirth'] = pd.to_datetime(drivers['dateOfBirth'])

    # Historic Filtering
    races_history = races[races['date'] <= race_date]
    race_to_predict = races_history[races_history['date'] == race_date].iloc[0]

    race_id_to_predict = race_to_predict['id']
    round_to_predict = int(race_to_predict['round'])

    race_results = race_results[race_results['raceId'] <= race_id_to_predict]
    qualifying = qualifying[qualifying['raceId'] <= race_id_to_predict]

    # Age and Qualifying
    drivers['driver_age'] = (race_date - drivers['dateOfBirth']) / pd.Timedelta(days=365.25)

    qualifying['reached_q1'] = qualifying['q1Millis'].notna()
    qualifying['reached_q2'] = qualifying['q2Millis'].notna()
    qualifying['reached_q3'] = qualifying['q3Millis'].notna()

    qualifying.rename(columns={
        'q1Millis': 'q1_millis',
        'q2Millis': 'q2_millis',
        'q3Millis': 'q3_millis',
        'positionDisplayOrder': 'position_qualifying'
    }, inplace=True)

    qualifying['q1_millis'] = qualifying['q1_millis'].fillna(0.0)
    qualifying['q2_millis'] = qualifying['q2_millis'].fillna(0.0)
    qualifying['q3_millis'] = qualifying['q3_millis'].fillna(0.0)

    # Merges and Feature Engineering
    race_results_past = race_results[race_results['raceId'] < race_id_to_predict].copy()

    race_results_past = pd.merge(
        left=race_results_past,
        right=races_history[['id', 'date', 'circuitId']],
        left_on='raceId',
        right_on='id',
        how='inner',
        validate='many_to_many'
    )

    race_results_past = race_results_past.sort_values(by='date')

    race_results_past['driver_momentum'] = (race_results_past.groupby('driverId')['positionDisplayOrder']
                                            .transform(lambda x: x.rolling(3, min_periods=1).mean()))

    past_in_circuit = race_results_past[race_results_past['circuitId'] == race_to_predict['circuitId']].copy()
    past_in_circuit['driver_track_affinity'] = past_in_circuit.groupby('driverId')[
        'positionDisplayOrder'].transform(lambda x: x.expanding(1).mean())

    race_results_past['is_dnf'] = race_results_past['positionNumber'].isna().astype(int)
    race_results_past['driver_dnf_rate'] = race_results_past.groupby('driverId')['is_dnf'].transform(
        lambda x: x.expanding(1).mean())
    race_results_past['constructor_dnf_rate'] = race_results_past.groupby('constructorId')['is_dnf'].transform(
        lambda x: x.expanding(1).mean())

    circuit_dnf_rate = race_results_past['is_dnf'].mean()

    team_race_totals = race_results_past.groupby(['constructorId', 'raceId', 'date'])[
        'positionDisplayOrder'].mean().reset_index().sort_values(by='date')
    team_race_totals['constructor_momentum'] = team_race_totals.groupby('constructorId')[
        'positionDisplayOrder'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    team_track_totals = past_in_circuit.groupby(['constructorId', 'raceId', 'date'])[
        'positionDisplayOrder'].mean().reset_index().sort_values(by='date')
    team_track_totals['constructor_track_affinity'] = team_track_totals.groupby('constructorId')[
        'positionDisplayOrder'].transform(lambda x: x.expanding(1).mean())

    driver_momentum_lookup = race_results_past.groupby('driverId')['driver_momentum'].last().reset_index()
    driver_track_lookup = past_in_circuit.groupby('driverId')['driver_track_affinity'].last().reset_index()
    driver_dnf_lookup = race_results_past.groupby('driverId')['driver_dnf_rate'].last().reset_index()

    constructor_momentum_lookup = team_race_totals.groupby('constructorId')[
        'constructor_momentum'].last().reset_index()
    constructor_track_lookup = team_track_totals.groupby('constructorId')[
        'constructor_track_affinity'].last().reset_index()
    constructor_dnf_lookup = race_results_past.groupby('constructorId')['constructor_dnf_rate'].last().reset_index()

    # Joining Data and Isolating the Target
    target_race_df = qualifying[qualifying['raceId'] == race_id_to_predict].copy()

    if target_race_df.empty:
        raise ValueError("There's no qualifying data for the chosen race. "
                         "Is F1DB already updated with Saturday's qualifying data?")

    target_race_df = pd.merge(target_race_df, starting_grid[['raceId', 'driverId', 'positionDisplayOrder']],
                              on=['raceId', 'driverId'], how='left', validate='many_to_many')
    target_race_df.rename(columns={'positionDisplayOrder': 'grid'}, inplace=True)

    target_race_df = pd.merge(target_race_df, drivers[['id', 'driver_age']], left_on='driverId', right_on='id',
                              how='left', validate='many_to_many')

    target_race_df = pd.merge(target_race_df, driver_momentum_lookup, on='driverId', how='left',
                              validate='many_to_many')
    target_race_df = pd.merge(target_race_df, driver_track_lookup, on='driverId', how='left', validate='many_to_many')
    target_race_df = pd.merge(target_race_df, driver_dnf_lookup, on='driverId', how='left', validate='many_to_many')

    target_race_df = pd.merge(target_race_df, constructor_momentum_lookup, on='constructorId', how='left',
                              validate='many_to_many')
    target_race_df = pd.merge(target_race_df, constructor_track_lookup, on='constructorId', how='left',
                              validate='many_to_many')
    target_race_df = pd.merge(target_race_df, constructor_dnf_lookup, on='constructorId', how='left',
                              validate='many_to_many')

    cols_fill = ['driver_momentum', 'driver_track_affinity', 'driver_dnf_rate', 'constructor_momentum',
                 'constructor_track_affinity', 'constructor_dnf_rate']
    target_race_df[cols_fill] = target_race_df[cols_fill].fillna(0.0)
    target_race_df['circuit_dnf_rate'] = float(circuit_dnf_rate) if pd.notna(circuit_dnf_rate) else 0.0

    target_race_df['circuitId'] = race_to_predict['circuitId']
    target_race_df['circuit_id_mapped'] = target_race_df['circuitId'].map(mappings['circuits'])
    target_race_df['constructor_id_mapped'] = target_race_df['constructorId'].map(mappings['constructors'])

    # Payload Builder
    grid_payload = []

    for _, data in target_race_df.iterrows():
        driver_obj = {
            'driver_ref': str(data['driverId']),
            'driver_age': float(data['driver_age']),
            'driver_momentum': float(data['driver_momentum']),
            'driver_track_affinity': float(data['driver_track_affinity']),
            'driver_dnf_rate': float(data['driver_dnf_rate']),

            'position_qualifying': int(data['position_qualifying']) if pd.notna(
                data['position_qualifying']) else 20,
            'grid': int(data['grid']) if pd.notna(data['grid']) else 20,
            'q1_millis': float(data['q1_millis']),
            'reached_q1': bool(data['reached_q1']),
            'q2_millis': float(data['q2_millis']),
            'reached_q2': bool(data['reached_q2']),
            'q3_millis': float(data['q3_millis']),
            'reached_q3': bool(data['reached_q3']),

            'constructor_id': int(data['constructor_id_mapped']) if pd.notna(data['constructor_id_mapped']) else None,
            'constructor_momentum': float(data['constructor_momentum']),
            'constructor_track_affinity': float(data['constructor_track_affinity']),
            'constructor_dnf_rate': float(data['constructor_dnf_rate']),

            'round': round_to_predict,
            'circuit_id': int(data['circuit_id_mapped']) if pd.notna(data['circuit_id_mapped']) else None,
            'circuit_dnf_rate': float(data['circuit_dnf_rate'])
        }
        grid_payload.append(DriverRaceData(**driver_obj))

    race_name = races[races['id'] == target_race_df['raceId'].iloc[0]]['officialName'].iloc[0]

    request = RacePredictionRequest(
        race_name=race_name,
        grid_data=grid_payload
    )

    return request
