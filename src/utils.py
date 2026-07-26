import pandas as pd


def load_X_and_y(self):
    data_dir = self.data_dir

    self.x_train = pd.read_csv(f'{data_dir}/X_train.csv')
    self.y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze('columns')
    self.x_test = pd.read_csv(f'{data_dir}/X_test.csv')
    self.y_test = pd.read_csv(f'{data_dir}/y_test.csv').squeeze('columns')
    self.train_years = pd.read_csv(f'{data_dir}/train_years.csv').squeeze('columns')

    categorical_cols = ['constructorId', 'circuitId']

    for col in categorical_cols:
        all_categories = pd.concat([self.x_train[col], self.x_test[col]]).unique()

        categorical_type = pd.CategoricalDtype(categories=all_categories, ordered=False)

        self.x_train[col] = self.x_train[col].astype(categorical_type)
        self.x_test[col] = self.x_test[col].astype(categorical_type)

    if len(self.train_years) != len(self.x_train):
        raise ValueError("Training years do not align with X_train. Rebuild processed data before training.")


def get_chronological_cv(self, validation_years=4):
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
