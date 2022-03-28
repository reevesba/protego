""" Random Forest
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.ensemble import AdaptiveRandomForestClassifier
from river.metrics import Accuracy
from river.drift import ADWIN


class AdaptiveRandomForest(Base):
    def __init__(self, n_models=10, max_features='sqrt', lambda_value=6,
                 metric=Accuracy, disable_weighted_vote=False,
                 drift_detector=ADWIN, warning_detector=ADWIN,
                 grace_period=50, max_depth=None, split_criterion='info_gain',
                 split_confidence=0.01, tie_threshold=0.05,
                 leaf_prediction='nba', nb_threshold=0,
                 nominal_attributes=None, splitter=None, binary_split=False,
                 max_size=32, memory_estimate_period=2000000,
                 stop_mem_management=False, remove_poor_attrs=False, 
                 merit_preprune=True, seed=None):
        self.model = AdaptiveRandomForestClassifier(
            n_models, max_features, lambda_value, metric,
            disable_weighted_vote, drift_detector,
            warning_detector, grace_period, max_depth,
            split_criterion, split_confidence, tie_threshold,
            leaf_prediction, nb_threshold, nominal_attributes, splitter,
            binary_split, max_size, memory_estimate_period,
            stop_mem_management, remove_poor_attrs, merit_preprune, seed
        )
        super().__init__(self.model)
