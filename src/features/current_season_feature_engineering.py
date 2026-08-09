import pandas as pd
from pandas import DataFrame, Series


CURRENT_SEASON_FEATURE_COLUMNS = [
    'current_season_points_per_race',
    'current_season_avg_finish',
    'current_season_podium_rate',
    'current_season_q3_rate',
    'current_constructor_points_per_race',
    'last_3_current_season_avg_finish',
]

F1_POINTS_BY_POSITION = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


def build_processed_current_season_features(
    features: DataFrame,
    y_position: Series,
    qid: Series,
    year: Series,
    driver_id: Series,
) -> DataFrame:
    source = features.copy()
    source['actual_position'] = y_position
    source['qid'] = qid
    source['year'] = year
    source['driverId'] = driver_id
    source['points'] = source['actual_position'].map(F1_POINTS_BY_POSITION).fillna(0.0)

    feature_rows = []
    for index, row in source.iterrows():
        prior = source[(source['year'] == row['year']) & (source['qid'] < row['qid'])]
        completed_races_count = prior['qid'].nunique()
        driver_prior = prior[prior['driverId'] == row['driverId']]
        constructor_prior = prior[prior['constructorId'] == row['constructorId']]

        feature_rows.append(
            _build_feature_row(
                driver_prior=driver_prior,
                constructor_prior=constructor_prior,
                completed_races_count=completed_races_count,
            )
        )

    return DataFrame(feature_rows, index=features.index)


def build_f1db_current_season_features(
    race_features: DataFrame,
    race_results: DataFrame,
    qualifying: DataFrame,
    race: Series,
) -> DataFrame:
    feature_rows = []
    prior_results = race_results[
        (race_results['year'] == int(race['year']))
        & (race_results['round'] < int(race['round']))
    ].copy()
    prior_qualifying = qualifying[
        (qualifying['year'] == int(race['year']))
        & (qualifying['round'] < int(race['round']))
    ].copy()
    completed_races_count = prior_results['raceId'].nunique()

    for index, row in race_features.iterrows():
        driver_prior = prior_results[prior_results['driverId'] == row['driverId']].copy()
        constructor_prior = prior_results[prior_results['constructorId'] == row['constructor_ref']].copy()
        driver_qualifying_prior = prior_qualifying[prior_qualifying['driverId'] == row['driverId']]

        if not driver_qualifying_prior.empty:
            q3_lookup = driver_qualifying_prior.set_index('raceId')['q3Millis'].notna()
            driver_prior['reached_q3'] = driver_prior['raceId'].map(q3_lookup).fillna(False)

        feature_rows.append(
            _build_feature_row(
                driver_prior=driver_prior,
                constructor_prior=constructor_prior,
                completed_races_count=completed_races_count,
            )
        )

    return pd.concat([race_features, DataFrame(feature_rows, index=race_features.index)], axis=1)


def _build_feature_row(
    driver_prior: DataFrame,
    constructor_prior: DataFrame,
    completed_races_count: int,
) -> dict:
    driver_points = driver_prior['points'].sum() if not driver_prior.empty else 0.0
    constructor_points = constructor_prior['points'].sum() if not constructor_prior.empty else 0.0
    divisor = completed_races_count or 1

    return {
        'current_season_points_per_race': float(driver_points / divisor),
        'current_season_avg_finish': _mean_or_zero(driver_prior, 'positionDisplayOrder', 'actual_position'),
        'current_season_podium_rate': _podium_rate(driver_prior),
        'current_season_q3_rate': _mean_bool_or_zero(driver_prior, 'reached_q3'),
        'current_constructor_points_per_race': float(constructor_points / divisor),
        'last_3_current_season_avg_finish': _last_3_avg_finish(driver_prior),
    }


def _mean_or_zero(data: DataFrame, primary_col: str, fallback_col: str) -> float:
    if data.empty:
        return 0.0

    col = primary_col if primary_col in data.columns else fallback_col
    return float(data[col].mean())


def _podium_rate(data: DataFrame) -> float:
    if data.empty:
        return 0.0

    position_col = 'positionDisplayOrder' if 'positionDisplayOrder' in data.columns else 'actual_position'
    return float((data[position_col] <= 3).mean())


def _mean_bool_or_zero(data: DataFrame, col: str) -> float:
    if data.empty or col not in data.columns:
        return 0.0

    return float(data[col].fillna(False).astype(bool).mean())


def _last_3_avg_finish(data: DataFrame) -> float:
    if data.empty:
        return 0.0

    sort_col = 'round' if 'round' in data.columns else 'qid'
    position_col = 'positionDisplayOrder' if 'positionDisplayOrder' in data.columns else 'actual_position'
    return float(data.sort_values(by=sort_col).tail(3)[position_col].mean())
