import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


class XGBFinishRacePredictor(object):
    def __init__(self, data_dir='../../data/03_processed/'):
        """Init the model and variables"""
        self.data_dir = data_dir
        self.model = XGBClassifier(n_jobs=-1, random_state=42, device='cuda',
                                   subsample=1, n_estimators=300, max_depth=2, learning_rate=0.075)
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.train_years = None

    def load_and_prepare_data(self):
        """Load train/test files already prepared by the notebooks."""
        self.x_train = pd.read_csv(f'{self.data_dir}X_train.csv')
        self.x_test = pd.read_csv(f'{self.data_dir}X_test.csv')
        self.y_train = pd.read_csv(f'{self.data_dir}y_train.csv').squeeze('columns')
        self.y_test = pd.read_csv(f'{self.data_dir}y_test.csv').squeeze('columns')
        self.train_years = pd.read_csv(f'{self.data_dir}train_years.csv').squeeze('columns')

        if len(self.train_years) != len(self.x_train):
            raise ValueError("Training years do not align with X_train. Rebuild processed data before tuning.")

    def _get_scale_pos_weight(self):
        """XGBoost positive class is mechanical DNF, so weight it by negative/positive ratio."""
        negative_count = np.count_nonzero(self.y_train == 0)
        positive_count = np.count_nonzero(self.y_train == 1)

        if positive_count == 0:
            raise ValueError("No positive mechanical DNF examples found in y_train.")

        return negative_count / positive_count

    def _get_chronological_cv(self, validation_years=4):
        """Expanding-window folds: train on past seasons, validate on one future season."""
        if self.train_years is None:
            raise ValueError("No training years available. Run load_and_prepare_data first.")

        years = sorted(self.train_years.unique())
        cv = []

        for validation_year in years[-validation_years:]:
            train_idx = self.train_years[self.train_years < validation_year].index.to_numpy()
            validation_idx = self.train_years[self.train_years == validation_year].index.to_numpy()

            if len(train_idx) == 0 or len(validation_idx) == 0:
                continue

            cv.append((train_idx, validation_idx))

        if not cv:
            raise ValueError("Could not build chronological CV folds.")

        return cv

    def train(self):
        """Train the model with the loaded data"""
        if self.x_train is None or self.x_test is None:
            raise ValueError("No training data. Run load_and_prepare_data first")

        self.model.set_params(scale_pos_weight=self._get_scale_pos_weight())
        self.model.fit(X=self.x_train, y=self.y_train)

    def tune_hyperparameters(self):
        """Execute the search for the best hyperparameters"""

        print("Start Tuning Hyperparameters")

        param_grid = {
            'max_depth': [2, 3, 4, 5, 8, 10],
            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'scale_pos_weight': [
                self._get_scale_pos_weight() * value
                for value in [0.5, 0.75, 1.0, 1.25, 1.5]
            ],
        }

        xgb_base = XGBClassifier(
            n_jobs=-1,
            random_state=42,
            device='cuda'
        )

        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_grid,
            n_iter=50,
            scoring='average_precision',
            cv=self._get_chronological_cv(),
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )

        random_search.fit(X=self.x_train, y=self.y_train)

        print("\n ===== Best hyperparameters: =====")
        print(random_search.best_params_)

        self.model = random_search.best_estimator_

    def evaluate(self):
        """Evaluate the model on the test data and return the metrics"""
        y_proba = self.model.predict_proba(X=self.x_test)

        mechanical_dnf_class_idx = np.nonzero(self.model.classes_ == 1)[0][0]
        proba_mechanical_dnf = y_proba[:, mechanical_dnf_class_idx]

        risk_threshold = 0.5

        custom_y_pred = np.where(proba_mechanical_dnf > risk_threshold, 1, 0)

        report = metrics.classification_report(self.y_test, custom_y_pred, zero_division=0)
        conf_matrix = metrics.confusion_matrix(self.y_test, custom_y_pred)

        print(f'Mechanical DNF average precision: {metrics.average_precision_score(self.y_test, proba_mechanical_dnf):.4f}')
        print(f'Mechanical DNF ROC AUC: {metrics.roc_auc_score(self.y_test, proba_mechanical_dnf):.4f}')
        print('===== Evaluation: Mechanical DNF Prediction =====')
        print(report)
        print('===== Confusion Matrix: Mechanical DNF Prediction =====')
        print(conf_matrix)


if __name__ == '__main__':
    predictor = XGBFinishRacePredictor()
    predictor.load_and_prepare_data()
    predictor.train()
    # predictor.tune_hyperparameters()
    predictor.evaluate()
