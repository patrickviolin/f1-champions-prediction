import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier


class XGBFinishRacePredictor(object):
    def __init__(self, data_dir='../../data/03_processed/'):
        """Init the model and variables"""
        self.data_dir = data_dir
        self.model = XGBClassifier(enable_categorical=True, n_jobs=-1, random_state=42, device='cuda')
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

    def evaluate(self):
        """Evaluate the model on the test data and return the metrics"""
        y_pred = self.model.predict(X=self.X_test)

        report = metrics.classification_report(self.y_test, y_pred)

        print('===== Evaluation: Finish the Race or DNF Prediction =====')
        print(report)
        return report


if __name__ == '__main__':
    predictor = XGBFinishRacePredictor()
    predictor.load_and_prepare_data()
    predictor.train()
    predictor.evaluate()
