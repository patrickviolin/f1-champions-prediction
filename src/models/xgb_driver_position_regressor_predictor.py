import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

import utils
from utils import load_x_and_y_data


class XGBDriverPositionRegressorPredictor(object):
    def __init__(self, data_dir='../../data/03_processed/'):
        """Init the model and variables"""
        self.model = XGBRegressor(
            random_state=42,
            device='cuda',
            n_jobs=-1,
            enable_categorical=True,
            max_depth=2,
            min_child_weight=1,
            learning_rate=0.025,
            n_estimators=205,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=2.0
        )
        self.data_dir = data_dir

        self.x_train, self.y_train, self.x_test, self.y_test, self.year_train = None, None, None, None, None
        self.qid_test = None

    def load_and_prepare_data(self):
        """Load train/test data used for training and testing"""
        load_x_and_y_data(self)

        # Keeping QID test to sort the ranking on evaluate
        self.qid_test = pd.read_csv(self.data_dir + 'qid_test.csv').squeeze('columns')

        # Keeping the same logic of ranker to ease comparing with ranker
        self.y_train = 25 - self.y_train
        self.y_test = 25 - self.y_test

    def train(self):
        """Train the model with XGBoost Regressor"""
        self.model.fit(
            self.x_train, self.y_train,
            eval_set=[(self.x_test, self.y_test)],
            verbose=False
        )

    def tune_hyperparameters(self, n_iter=50):
        """Tune hyperparameters"""

        print("Start Tuning Hyperparameters")

        param_grid = {
            'max_depth': [2, 3, 4, 5, 8, 10],
            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7, 10],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 5.0],
        }

        search_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            random_state=42,
            n_iter=n_iter,
            n_jobs=-1,
            verbose=2,
            scoring='neg_root_mean_squared_error',
            cv=utils.get_chronological_cv(self, 4)
        )

        search_cv.fit(self.x_train, self.y_train)

        print('\nHyperparameters tuned for XGBoost Regressor')
        print(f'Best Params: {search_cv.best_params_}')

        self.model = search_cv.best_estimator_

    def predict(self):
        """Predict with XGBoost Regressor"""
        self.model.predict(self.x_test)


if __name__ == '__main__':
    predictor = XGBDriverPositionRegressorPredictor()
    predictor.load_and_prepare_data()
    predictor.train()
    # predictor.tune_hyperparameters()
    utils.evaluate_xgboost(predictor)
    utils.show_feature_importance(predictor)
    utils.save_model(predictor, file_name='xgb_regressor_model.json')
