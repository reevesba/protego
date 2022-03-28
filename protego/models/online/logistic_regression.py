""" Logistic Regression
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.linear_model import LogisticRegression


class LogisticRegressionClassifier(Base):
    def __init__(self, optimizer=None, loss=None, l2=0.0,
                 intercept_init=0.0, intercept_lr=0.01,
                 clip_gradient=1000000000000.0, initializer=None):
        self.model = LogisticRegression(
            optimizer, loss, l2, intercept_init,
            intercept_lr, clip_gradient, initializer
        )
        super().__init__(self.model)
