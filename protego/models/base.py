""" Base Classifiers
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "MIT"

from river.stream import iter_pandas
from protego.models import types as t
from pandas import DataFrame, Series
import numpy as np
import pickle
import os


class Base:
    def __init__(self: t.BaseT, model: t.ModelT) -> None:
        """ Initialize Base class
            Parameters
            ----------
            self: Base instance
            model: Model instance

            Returns
            -------
            None
        """
        self.model = model
        self.dir, _ = os.path.split(os.path.split(__file__)[0])
        self.dir = self.dir + '/pretrained/'

    def save(self: t.BaseT, filename: str) -> None:
        """ Save a model to file
            Parameters
            ----------
            self: Base instance
            filename: File name for saving

            Returns
            -------
            None
        """
        with open(self.dir + filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self: t.BaseT, filename: str) -> None:
        """ Load a model from file
            Parameters
            ----------
            self: Base instance
            filename: File name for loading

            Returns
            -------
            None
        """
        with open(self.dir + filename, 'rb') as f:
            self.model = pickle.load(f)


class BaseBatch(Base):
    def __init__(self: t.BaseBatchT, model: t.BatchModelT) -> None:
        """ Initialize BaseBatch class
            Parameters
            ----------
            self: BaseBatch instance
            model: Model instance

            Returns
            -------
            None
        """
        super().__init__(model)

    def train(self: t.BaseBatchT, train_data: DataFrame,
              train_labels: Series) -> None:
        """ Train a batch model
            Parameters
            ----------
            self: BaseBatch instance
            train_data: Samples to train
            train_labels: Sample Classes

            Returns
            -------
            None
        """
        self.model.fit(train_data, train_labels)

    def predict(self: t.BaseBatchT, data: DataFrame) -> np.array:
        """ Predict samples with batch model
            Parameters
            ----------
            self: BaseBatch instance
            data: Samples to predict

            Returns
            -------
            predictions: numpy array
        """
        return self.model.predict(data)


class BaseOnline(Base):
    def __init__(self: t.BaseOnlineT, model: t.OnlineModelT) -> None:
        """ Initialize BaseOnline class
            Parameters
            ----------
            self: BaseOnline instance
            model: Model instance

            Returns
            -------
            None
        """
        super().__init__(model)

    def train(self: t.BaseOnlineT, train_data: DataFrame,
              train_labels: Series) -> None:
        """ Train an online model
            Parameters
            ----------
            self: BaseOnline instance
            train_data: Samples to train
            train_labels: Sample Classes

            Returns
            -------
            None
        """
        for X, y in iter_pandas(X=train_data, y=train_labels):
            self.model = self.model.learn_one(X, y)

    def predict(self: t.BaseOnlineT, data: DataFrame) -> np.array:
        """ Predict samples with online model
            Parameters
            ----------
            self: BaseOnline instance
            data: Samples to predict

            Returns
            -------
            predictions: numpy array
        """
        predictions = np.array([])
        for X, _ in iter_pandas(X=data):
            prediction = self.model.predict_one(X)
            predictions = np.append(predictions, prediction)
        return predictions
