import pandas as pd

from models.xgb_driver_position_ranker_predictor import XGBDriverPositionRankerPredictor


class F1ChampionshipSimulator:
    def __init__(self, target_year=2024):
        self.target_year = target_year

        self.points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

        self.predictor = XGBDriverPositionRankerPredictor()
        self.predictor.load_and_prepare_data()
        self.predictor.train()

        data_dir = '../../data/03_processed/'

        self.driver_test = pd.read_csv(f'{data_dir}driver_test.csv').squeeze('columns')
        self.year_test = pd.read_csv(f'{data_dir}year_test.csv').squeeze('columns')

    def simulate_season(self):
        """Simulate F1 season"""
        print(f'Starting simulation of {self.target_year}\'s F1 season')

        y_pred_scores = self.predictor.model.predict(self.predictor.x_test)

        drivers = pd.read_csv('../../data/01_raw/drivers.csv')

        drivers = drivers[['driverId', 'driverRef']]

        results = pd.DataFrame({
            'year': self.year_test,
            'race_id': self.predictor.qid_test,
            'driver_id': self.driver_test,
            'actual_position': 25 - self.predictor.y_test,
            'predicted_score': y_pred_scores
        })

        results = pd.merge(results, drivers, left_on='driver_id', right_on='driverId', how='inner',
                           validate='many_to_one')

        results.rename(columns={'driverRef': 'driver_ref'}, inplace=True)

        results = results[results['year'] == self.target_year].copy()

        results = results.sort_values(by=['race_id', 'predicted_score'], ascending=[True, False])

        results['predicted_position'] = results.groupby('race_id').cumcount() + 1

        results['predicted_points'] = results['predicted_position'].map(self.points_system).fillna(0)
        results['actual_points'] = results['actual_position'].map(self.points_system).fillna(0)

        championship = results.groupby(['driver_id', 'driver_ref'])[['predicted_points', 'actual_points']].sum()

        championship = championship.sort_values(by='predicted_points', ascending=False).reset_index()

        print("=" * 45)
        print(f'{self.target_year} CHAMPIONSHIP CLASSIFICATION')
        print("=" * 45)
        print('     Driver     | Predicted Score | Actual Score | Status ')
        print('-' * 45)

        for _, row in championship.iterrows():
            d_ref = row['driver_ref']
            p_score = int(row['predicted_points'])
            a_score = int(row['actual_points'])

            if p_score > a_score:
                status = "Optimistic"
            elif p_score < a_score:
                status = "Pessimistic"
            else:
                status = "Exact"

            print(f"{d_ref:^15} | {p_score:^13} | {a_score:^9} | {status}")


if __name__ == '__main__':
    simulator = F1ChampionshipSimulator()
    simulator.simulate_season()
