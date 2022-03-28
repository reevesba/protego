""" Decision Tree
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Base):
    def __init__(self, criterion='gini', splitter='best',
                 max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None,
                 max_leaf_node=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0):
        self.model = DecisionTreeClassifier(
            criterion, splitter, max_depth, min_samples_split,
            min_samples_leaf, min_weight_fraction_leaf, max_features,
            random_state, max_leaf_node, min_impurity_decrease,
            class_weight, ccp_alpha
        )
        super().__init__(self.model)
