PRE_QUALIFYING_EXCLUDED_COLUMNS = [
    'position_qualifying',
    'q1_millis',
    'reached_q1',
    'q2_millis',
    'reached_q2',
    'q3_millis',
    'reached_q3',
    'grid',
]

PRE_QUALIFYING_FEATURE_ORDER = [
    'driver_age',
    'driver_momentum',
    'current_season_points_per_race',
    'current_season_avg_finish',
    'current_season_podium_rate',
    'current_season_q3_rate',
    'constructorId',
    'constructor_momentum',
    'current_constructor_points_per_race',
    'round',
    'circuitId',
    'last_3_current_season_avg_finish',
    'driver_track_affinity',
    'constructor_track_affinity',
    'constructor_dnf_rate',
    'driver_dnf_rate',
    'circuit_dnf_rate',
]

PRE_QUALIFYING_CATEGORICAL_COLUMNS = ['constructorId', 'circuitId']
