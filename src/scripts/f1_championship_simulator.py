import pandas as pd

from models.xgb_driver_position_regressor_predictor import XGBDriverPositionRegressorPredictor


class F1ChampionshipSimulator:
    def __init__(self, target_year=2023):
        self.target_year = target_year

        self.points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

        self.predictor = XGBDriverPositionRegressorPredictor()
        self.predictor.load_and_prepare_data()
        self.predictor.train()

        data_dir = '../../data/03_processed/'

        self.driver_test = pd.read_csv(f'{data_dir}driver_test.csv').squeeze('columns')
        self.year_test = pd.read_csv(f'{data_dir}year_test.csv')

    def simulate_season(self):
        """Simulate F1 season"""
        print(f'Starting simulation of {self.target_year}\'s F1 season')

        y_pred_scores = self.predictor.model.predict(self.predictor.x_test)

        results = pd.DataFrame({
            'year': self.year_test,
            'race_id': self.predictor.qid_test,
            'driver_id': self.driver_test,
            'actual_position': 25 - self.predictor.y_test,
            'predicted_score': y_pred_scores
        })

        results = results[['year'] == self.target_year]

        