""" Random Forest
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.ensemble import AdaptiveRandomForestClassifier


class AdaptiveRandomForest(Base):
    def __init__(self, ):
        self.model = AdaptiveRandomForestClassifier(
            grace_period, max_depth, split_criterion, split_confidence,
            tie_threshold, leaf_prediction, nb_threshold, nominal_attributes,
            splitter, bootstrap_sampling, drift_window_threshold,
            adwin_confidence, binary_split, max_size, memory_estimate_period,
            stop_mem_management, remove_poor_attrs, merit_prune, seed
        )
        super().__init__(self.model)
