""" KNN w/ ADWIN
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.neighbors import KNNADWINClassifier


class KNNADWIN(Base):
    def __init__(self, n_neighbors=5, window_size=1000,
                 leaf_size=30, p=2):
        self.model = KNNADWINClassifier(
            n_neighbors, window_size, leaf_size, p
        )
        super().__init__(self.model)
