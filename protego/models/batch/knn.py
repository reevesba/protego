""" KNN
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from sklearn.neighbors import KNeighborsClassifier


class KNeighbors(Base):
    def __init__(self, n_neighbors=5, weights='uniform',
                 algorithm='auto', leaf_size=30, p=2,
                 metric='minkowski', metric_params=None, n_jobs=None):
        self.model = KNeighborsClassifier(
            n_neighbors, weights, algorithm,
            leaf_size, p, metric, metric_params, n_jobs
        )
        super().__init__(self.model)
