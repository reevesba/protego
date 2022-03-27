""" Online Random Forest
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "February 12, 2022"
__license__ = "None"

from river import AdaptiveRandomForestClassifier
from river.stream import iter_pandas
from river.metrics import Accuracy
import pandas as pd
import pickle


class AdaptiveRandomForest:
    def __init__(self):
        self.model = AdaptiveRandomForestClassifier()

    def save(self, file):
        with open("../pretrained/" + file + ".pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file):
        with open("../pretrained/" + file + ".pkl", "wb") as f:
            self.model = pickle.load(f)

    def train(self, data_file, print_every=None):
        dataset = pd.read_csv("../data/" + data_file + ".csv")
        cols = dataset.columns[:-1]
        datastream = iter_pandas(X=dataset[cols], y=dataset['label'])
        acc = Accuracy()

        if print_every is None:
            print_every = len(dataset)

        # Do the training
        index = 0
        for X, y in datastream:
            y_pred = self.model.predict_one(X)
            acc = acc.update(y, y_pred)
            if index % print_every == 0:
                print(str(index) + ": " + str(acc))

    def predict(self, payload, do_update=False):
        return self.model.predict_one(payload)
