from xgboost import XGBRegressor


class XGBDriverPositionRankerPredictor(object):
    def __init__(self, data_dir='../../data/03_processed/'):
        """Init the model and variables"""
        self.model = XGBRegressor()

        self.x_train, self.y_train, self.x_test, self.y_train, = None, None, None, None

    def load_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

if __name__ == '__main__':
    predictor = XGBDriverPositionRankerPredictor()
    predictor.__init__()