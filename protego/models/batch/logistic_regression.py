""" Logistic Regression
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(Base):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False,
                 n_jobs=None, l1_ratio=None):
        self.model = LogisticRegression(
            penalty, dual, tol, C, fit_intercept, intercept_scaling,
            class_weight, random_state, solver, max_iter,
            multi_class, verbose, warm_start, n_jobs, l1_ratio
        )
        super().__init__(self.model)
