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
    'constructorId',
    'constructor_momentum',
    'round',
    'circuitId',
    'driver_track_affinity',
    'constructor_track_affinity',
    'constructor_dnf_rate',
    'driver_dnf_rate',
    'circuit_dnf_rate',
]

PRE_QUALIFYING_CATEGORICAL_COLUMNS = ['constructorId', 'circuitId']
