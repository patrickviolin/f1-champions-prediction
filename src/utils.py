import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ndcg_score
from xgboost import plot_importance


def load_x_and_y_data(self):
    data_dir = self.data_dir

    self.x_train = pd.read_csv(f'{data_dir}/X_train.csv')
    self.y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze('columns')
    self.x_test = pd.read_csv(f'{data_dir}/X_test.csv')
    self.y_test = pd.read_csv(f'{data_dir}/y_test.csv').squeeze('columns')
    self.year_train = pd.read_csv(f'{data_dir}/year_train.csv').squeeze('columns')

    categorical_cols = ['constructorId', 'circuitId']

    for col in categorical_cols:
        all_categories = pd.concat([self.x_train[col], self.x_test[col]]).unique()

        categorical_type = pd.CategoricalDtype(categories=all_categories, ordered=False)

        self.x_train[col] = self.x_train[col].astype(categorical_type)
        self.x_test[col] = self.x_test[col].astype(categorical_type)

    if len(self.year_train) != len(self.x_train):
        raise ValueError("Training years do not align with X_train. Rebuild processed data before training.")


def evaluate_xgboost(self):
    if self.x_test is None or self.y_test is None:
        raise ValueError("No test data. Run load_and_prepare_data first")

    y_pred_scores = self.model.predict(self.x_test)

    results = pd.DataFrame({
        'race_id': self.qid_test,
        'actual_relevance': self.y_test,
        'predicted_relevance': y_pred_scores,
    })

    results['actual_position'] = 25 - results['actual_relevance']

    ndcg_list = []
    for race_id, group in results.groupby('race_id'):

        if len(group) > 1:
            score = ndcg_score([group['actual_relevance'].values], [group['predicted_relevance'].values])
            ndcg_list.append(score)

    avg_ndcg = np.mean(ndcg_list)

    print('\n' + '=' * 50)
    print('RANKING EVAL')
    print('=' * 50)
    print(f'NDCG Global Mean: {avg_ndcg:.4f} (From 0.0 to 1.0)')
    print('-' * 50)

    rng = np.random.default_rng(42)

    race_example = results['race_id'].iloc[rng.integers(0, len(results))]
    race_table = results[results['race_id'] == race_example].copy()

    race_table = race_table.sort_values('predicted_relevance', ascending=False).reset_index(drop=True)
    race_table['predicted_position'] = race_table.index + 1

    print(f'\nRace Simulation (race_id: {race_example})')
    print('Predicted | Real | Mathematical Score')
    print('-' * 35)

    for _, row in race_table.iterrows():
        prev = int(row['predicted_position'])
        real = int(row['actual_position'])
        score = row['predicted_relevance']

        print(f'{prev:^8} | {real:^4} | {score:>.4f}')


def get_chronological_cv(self, validation_years=4):
    """Expanding-window folds: train on past seasons, validate on one future season."""
    if self.year_train is None:
        raise ValueError("No training years available. Run load_and_prepare_data first.")

    years = sorted(self.year_train.unique())
    cv = []

    for validation_year in years[-validation_years:]:
        train_idx = self.year_train[self.year_train < validation_year].index.to_numpy()
        validation_idx = self.year_train[self.year_train == validation_year].index.to_numpy()

        if len(train_idx) == 0 or len(validation_idx) == 0:
            continue

        cv.append((train_idx, validation_idx))

    if not cv:
        raise ValueError("Could not build chronological CV folds.")

    return cv


def show_feature_importance(self):
    """Extracts and plot the importance of trained model features."""

    if self.model is None:
        raise ValueError("The model was still not trained. Run train() first.")

    print('\n' + '=' * 50)
    print('GENERATING FEATURE IMPORTANCE PLOT')
    print('=' * 50)

    # Criamos o quadro da figura
    _, ax = plt.subplots(figsize=(12, 8))

    # importance_type='gain' is the most reliable metric
    plot_importance(
        self.model,
        ax=ax,
        importance_type='gain',
        max_num_features=20,  # Show Top 20
        height=0.6,
        title='F1 Feature Importance (Metric: Gain)',
        xlabel='Average Information Gain',
        ylabel='Variables (Features)'
    )

    plt.tight_layout()
    plt.show()
