""" Perceptron
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from sklearn.linear_model import Perceptron


class PerceptronClassifier(Base):
    def __init__(self, penalty=None, alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, eta0=1, n_jobs=None,
                 random_state=None, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5,
                 class_weight=None, warm_start=False):
        self.model = Perceptron(
            penalty, alpha, l1_ratio, fit_intercept, max_iter,
            tol, shuffle, verbose, eta0, n_jobs, random_state,
            early_stopping, validation_fraction, n_iter_no_change,
            class_weight, warm_start
        )
        super().__init__(self.model)
