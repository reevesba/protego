""" Random Forest
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Base):
    def __init__(self, n_estimators=100, criterion='gini',
                 max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, bootstrap=True,
                 oob_score=False, n_jobs=None, random_state=None,
                 verbose=0, warm_start=False, class_weight=None,
                 ccp_alpha=0.0, max_samples=None):
        self.model = RandomForestClassifier(
            n_estimators, criterion, max_depth, min_samples_split,
            min_samples_leaf, min_weight_fraction_leaf, max_features,
            max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score,
            n_jobs, random_state, verbose, warm_start, class_weight,
            ccp_alpha, max_samples
        )
        super().__init__(self.model)
