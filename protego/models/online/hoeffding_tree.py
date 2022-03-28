""" Hoeffding Tree
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.tree import HoeffdingAdaptiveTreeClassifier


class HoeffdingAdaptiveTree(Base):
    def __init__(self, grace_period=200, max_depth=None,
                 split_criterion='info_gain', split_confidence=1e-07,
                 tie_threshold=0.05, leaf_prediction='nba', nb_threshold=0,
                 nominal_attributes=None, splitter=None,
                 bootstrap_sampling=True, drift_window_threshold=300,
                 adwin_confidence=0.002, binary_split=False,
                 max_size=100, memory_estimate_period=1000000,
                 stop_mem_management=False, remove_poor_attrs=False,
                 merit_prune=True, seed=None):
        self.model = HoeffdingAdaptiveTreeClassifier(
            grace_period, max_depth, split_criterion, split_confidence,
            tie_threshold, leaf_prediction, nb_threshold, nominal_attributes,
            splitter, bootstrap_sampling, drift_window_threshold,
            adwin_confidence, binary_split, max_size, memory_estimate_period,
            stop_mem_management, remove_poor_attrs, merit_prune, seed
        )
        super().__init__(self.model)
