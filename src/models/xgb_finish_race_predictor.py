import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


class XGBFinishRacePredictor(object):
    def __init__(self, data_dir='../../data/03_processed/'):
        """Init the model and variables"""
        self.data_dir = data_dir
        self.model = XGBClassifier(enable_categorical=True, n_jobs=-1, random_state=42, device='cuda',
                                   subsample=0.5, scale_pos_weight=10, n_estimators=300, max_depth=10, learning_rate=0.2)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_prepare_data(self):
        """Load the CSV files and applies the necessary preprocessing"""
        self.X_train = pd.read_csv(f'{self.data_dir}X_train.csv')
        self.X_test = pd.read_csv(f'{self.data_dir}X_test.csv')
        self.y_train = pd.read_csv(f'{self.data_dir}y_train.csv')
        self.y_test = pd.read_csv(f'{self.data_dir}y_test.csv')

        categorical_cols = ['constructorId', 'circuitId']

        for col in categorical_cols:
            all_categories = pd.concat([self.X_train[col], self.X_test[col]]).unique()

            categorical_type = pd.CategoricalDtype(categories=all_categories, ordered=False)

            self.X_train[col] = self.X_train[col].astype(categorical_type)
            self.X_test[col] = self.X_test[col].astype(categorical_type)

    def train(self):
        """Train the model with the loaded data"""
        if self.X_train is None or self.X_test is None:
            raise ValueError("No training data. Run load_and_prepare_data first")

        self.model.fit(X=self.X_train, y=self.y_train)

    def tune_hyperparameters(self):
        """Execute the search for the best hyperparameters"""

        print("Start Tuning Hyperparameters")

        param_grid = {
            'max_depth': [2, 3, 4, 5, 8, 10],
            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'scale_pos_weight': [2, 3, 5, 7, 10],
        }

        xgb_base = XGBClassifier(
            enable_categorical=True,
            n_jobs=-1,
            random_state=42,
            device='cuda'
        )

        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_grid,
            n_iter=50,
            scoring='f1_macro',
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )

        random_search.fit(X=self.X_train, y=self.y_train)

        print("\n ===== Best hyperparameters: =====")
        print(random_search.best_params_)

        self.model = random_search.best_estimator_

    def evaluate(self):
        """Evaluate the model on the test data and return the metrics"""
        y_proba = self.model.predict_proba(X=self.X_test)

        proba_dnf = y_proba[:, 0]

        risk_threshold = 0.15

        custom_y_pred = np.where(proba_dnf > risk_threshold, 1, 0)

        report = metrics.classification_report(self.y_test, custom_y_pred)
        conf_matrix = metrics.confusion_matrix(self.y_test, custom_y_pred)

        print('===== Evaluation: Finish the Race or DNF Prediction =====')
        print(report)
        print('===== Confusion Matrix: Finish the Race or DNF Prediction =====')
        print(conf_matrix)


if __name__ == '__main__':
    predictor = XGBFinishRacePredictor()
    predictor.load_and_prepare_data()
    predictor.train()
    # predictor.tune_hyperparameters()
    predictor.evaluate()
