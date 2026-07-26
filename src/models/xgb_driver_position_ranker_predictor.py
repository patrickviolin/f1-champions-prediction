import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ndcg_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRanker, plot_importance

from utils import get_chronological_cv, load_X_and_y


class XGBDriverPositionRankerPredictor(object):
    def __init__(self, data_dir='../../data/03_processed/'):
        """Init the model and variables"""
        self.data_dir = data_dir

        self.objective = 'rank:ndcg'
        self.eval_metric = 'ndcg'

        self.model = XGBRanker(
            enable_categorical=True,
            n_jobs=-1,
            random_state=42,
            device='cuda',
            max_depth=4,
            min_child_weight=7,
            learning_rate=0.025,
            n_estimators=300,
            subsample=0.5,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective=self.objective,
            eval_metric=self.eval_metric
        )
        self.x_train, self.x_test, self.y_train, self.y_test, self.qid_train, self.qid_test = None, None, None, None, None, None
        self.train_years = None

    def load_and_prepare_data(self):
        """Load train/test files already prepared by the notebooks."""
        load_X_and_y(self)

        self.qid_train = pd.read_csv(f'{self.data_dir}qid_train.csv').squeeze('columns')
        self.qid_test = pd.read_csv(f'{self.data_dir}qid_test.csv').squeeze('columns')

        self.y_train = 25 - self.y_train
        self.y_test = 25 - self.y_test

    @staticmethod
    def _mean_ndcg_by_race(y_true, y_score, qid):
        """Calculate mean NDCG by race/query group."""
        results = pd.DataFrame({
            'qid': qid,
            'y_true': y_true,
            'y_score': y_score,
        })

        ndcg_list = []
        for _, group in results.groupby('qid', sort=False):
            if len(group) > 1:
                score = ndcg_score([group['y_true'].values], [group['y_score'].values])
                ndcg_list.append(score)

        if not ndcg_list:
            raise ValueError("No valid race groups available to calculate NDCG.")

        return float(np.mean(ndcg_list))

    def train(self):
        """Train the model with the loaded data"""
        if self.x_train is None or self.x_test is None:
            raise ValueError("No training data. Run load_and_prepare_data first")

        self.model.fit(X=self.x_train, y=self.y_train, qid=self.qid_train, eval_qid=[self.qid_test],
                       eval_set=[(self.x_test, self.y_test)])

    def tune_hyperparameters(self, n_iter=50):
        """Tune XGBRanker with chronological CV and per-race NDCG scoring."""

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

        cv = get_chronological_cv(self)
        sampled_params = list(ParameterSampler(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=42,
        ))

        search_results = []
        best_score = -np.inf
        best_params = None

        for i, params in enumerate(sampled_params, start=1):
            fold_scores = []

            print(f'\n[{i}/{len(sampled_params)}] Testing params: {params}')

            for fold, (train_idx, validation_idx) in enumerate(cv, start=1):
                x_train_fold = self.x_train.iloc[train_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                qid_train_fold = self.qid_train.iloc[train_idx]

                x_validation_fold = self.x_train.iloc[validation_idx]
                y_validation_fold = self.y_train.iloc[validation_idx]
                qid_validation_fold = self.qid_train.iloc[validation_idx]

                model = XGBRanker(
                    enable_categorical=True,
                    n_jobs=-1,
                    random_state=42,
                    device='cuda',
                    objective=self.objective,
                    eval_metric=self.eval_metric,
                    **params,
                )

                model.fit(
                    X=x_train_fold,
                    y=y_train_fold,
                    qid=qid_train_fold,
                    eval_set=[(x_validation_fold, y_validation_fold)],
                    eval_qid=[qid_validation_fold],
                    verbose=False,
                )

                validation_scores = model.predict(x_validation_fold)
                fold_score = self._mean_ndcg_by_race(
                    y_true=y_validation_fold,
                    y_score=validation_scores,
                    qid=qid_validation_fold,
                )
                fold_scores.append(fold_score)

                print(f'  Fold {fold} NDCG: {fold_score:.4f}')

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            search_results.append({
                'mean_ndcg': mean_score,
                'std_ndcg': std_score,
                **params,
            })

            print(f'  Mean NDCG: {mean_score:.4f} (+/- {std_score:.4f})')

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print("\n ===== Best hyperparameters: =====")
        print(best_params)
        print(f'Best CV NDCG: {best_score:.4f}')

        self.model = XGBRanker(
            enable_categorical=True,
            n_jobs=-1,
            random_state=42,
            device='cuda',
            objective=self.objective,
            eval_metric=self.eval_metric,
            **best_params,
        )

        self.model.fit(
            X=self.x_train,
            y=self.y_train,
            qid=self.qid_train,
            eval_set=[(self.x_test, self.y_test)],
            eval_qid=[self.qid_test],
            verbose=False,
        )

        return pd.DataFrame(search_results).sort_values('mean_ndcg', ascending=False)

    def evaluate(self):
        """Evaluate the model on the test data and return the metrics"""

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

        race_example = results['race_id'].iloc[np.random.randint(0, len(results) - 1)]
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


if __name__ == '__main__':
    predictor = XGBDriverPositionRankerPredictor()
    predictor.load_and_prepare_data()
    predictor.train()
    # predictor.tune_hyperparameters()
    predictor.evaluate()
    predictor.show_feature_importance()
