""" Base Batch Classifier
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"


class Base:
    def __init__(self, model):
        self.model = model

    def train(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
