""" Base Classifiers
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from river.stream import iter_pandas
import pickle
import os


class Base:
    def __init__(self, model):
        self.model = model
        self.dir, _ = os.path.split(os.path.split(__file__)[0])
        self.dir = self.dir + '/pretrained/'

    def save(self, filename):
        with open(self.dir + filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        with open(self.dir + filename, 'rb') as f:
            self.model = pickle.load(f)


class BaseBatch(Base):
    def __init__(self, model):
        super().__init__(model)

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, data):
        return self.model.predict(data)


class BaseOnline(Base):
    def __init__(self, model):
        super().__init__(model)

    def train(self, train_data, train_labels):
        for X, y in iter_pandas(X=train_data, y=train_labels):
            self.model = self.model.learn_one(X, y)

    def predict(self, data):
        return self.model.predict_one(data)
