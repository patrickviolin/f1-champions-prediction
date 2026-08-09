from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series
from xgboost import XGBRanker

from ml.pre_qualifying_features import (
    PRE_QUALIFYING_EXCLUDED_COLUMNS,
    PRE_QUALIFYING_FEATURE_ORDER,
)
from utils import ml_utils


def _to_pre_qualifying_features(features: pd.DataFrame) -> DataFrame | Series:
    return features.drop(columns=PRE_QUALIFYING_EXCLUDED_COLUMNS)[PRE_QUALIFYING_FEATURE_ORDER]


class XGBPreQualifyingRankerPredictor:
    def __init__(self, data_dir: Path | None = None):
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = data_dir or project_root / 'train_and_test_data' / '03_processed'
        self.model_path = project_root / 'models' / 'xgb_pre_qualifying_ranker_model.json'
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
            objective='rank:ndcg',
            eval_metric='ndcg',
        )
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.qid_train = None
        self.qid_test = None
        self.year_train = None

    def load_and_prepare_data(self) -> None:
        ml_utils.load_x_and_y_data(self)

        self.x_train = _to_pre_qualifying_features(self.x_train)
        self.x_test = _to_pre_qualifying_features(self.x_test)

        self.qid_train = pd.read_csv(f'{self.data_dir}/qid_train.csv').squeeze('columns')
        self.qid_test = pd.read_csv(f'{self.data_dir}/qid_test.csv').squeeze('columns')

        self.y_train = 25 - self.y_train
        self.y_test = 25 - self.y_test

    def train(self) -> None:
        if self.x_train is None or self.x_test is None:
            raise ValueError('No training data. Run load_and_prepare_data first')

        self.model.fit(
            X=self.x_train,
            y=self.y_train,
            qid=self.qid_train,
            eval_qid=[self.qid_test],
            eval_set=[(self.x_test, self.y_test)],
            verbose=False,
        )


if __name__ == '__main__':
    predictor = XGBPreQualifyingRankerPredictor()
    predictor.load_and_prepare_data()
    predictor.train()
    ml_utils.evaluate_xgboost(predictor)
    predictor.model.save_model(predictor.model_path)
    print(f'Model saved in {predictor.model_path}')
