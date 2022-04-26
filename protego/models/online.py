""" Online Models

    1. Hoeffding Adaptive Tree
    2. Adaptive K-Nearest Neighbors
    3. Logistic Regression
    4. Naive Bayes
    5. Perceptron
    6. Random Forest
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "MIT"

from protego.models.base import BaseOnline
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.neighbors import KNNADWINClassifier
from river.linear_model import LogisticRegression
from river.linear_model import Perceptron
from river.naive_bayes import MultinomialNB
from river.ensemble import AdaptiveRandomForestClassifier
from river.metrics import Accuracy
from river.drift import ADWIN


class HoeffdingAdaptiveTree(BaseOnline):
    def __init__(
        self,
        grace_period=200,
        max_depth=None,
        split_criterion="info_gain",
        split_confidence=1e-07,
        tie_threshold=0.05,
        leaf_prediction="nba",
        nb_threshold=0,
        nominal_attributes=None,
        splitter=None,
        bootstrap_sampling=True,
        drift_window_threshold=300,
        adwin_confidence=0.002,
        binary_split=False,
        max_size=100,
        memory_estimate_period=1000000,
        stop_mem_management=False,
        remove_poor_attrs=False,
        merit_preprune=True,
        seed=None
    ):
        super().__init__(
            HoeffdingAdaptiveTreeClassifier(
                grace_period=grace_period,
                max_depth=max_depth,
                split_criterion=split_criterion,
                split_confidence=split_confidence,
                tie_threshold=tie_threshold,
                leaf_prediction=leaf_prediction,
                nb_threshold=nb_threshold,
                nominal_attributes=nominal_attributes,
                splitter=splitter,
                bootstrap_sampling=bootstrap_sampling,
                drift_window_threshold=drift_window_threshold,
                adwin_confidence=adwin_confidence,
                binary_split=binary_split,
                max_size=max_size,
                memory_estimate_period=memory_estimate_period,
                stop_mem_management=stop_mem_management,
                remove_poor_attrs=remove_poor_attrs,
                merit_preprune=merit_preprune,
                seed=seed
            )
        )


class KNNADWIN(BaseOnline):
    def __init__(
        self,
        n_neighbors=5,
        window_size=1000,
        leaf_size=30,
        p=2
    ):
        super().__init__(
            KNNADWINClassifier(
                n_neighbors=n_neighbors,
                window_size=window_size,
                leaf_size=leaf_size,
                p=p
            )
        )


class LogisticRegressionClassifier(BaseOnline):
    def __init__(
        self,
        optimizer=None,
        loss=None,
        l2=0.0,
        intercept_init=0.0,
        intercept_lr=0.01,
        clip_gradient=1000000000000.0,
        initializer=None
    ):
        super().__init__(
            LogisticRegression(
                optimizer=optimizer,
                loss=loss,
                l2=l2,
                intercept_init=intercept_init,
                intercept_lr=intercept_lr,
                clip_gradient=clip_gradient,
                initializer=initializer
            )
        )


class NaiveBayes(BaseOnline):
    def __init__(
        self,
        alpha=1.0
    ):
        super().__init__(
            MultinomialNB(
                alpha=alpha
            )
        )


class PerceptronClassifier(BaseOnline):
    def __init__(
        self,
        l2=0.0,
        clip_gradient=1000000000000.0,
        initializer=None
    ):
        super().__init__(
            Perceptron(
                l2=l2,
                clip_gradient=clip_gradient,
                initializer=initializer
            )
        )


class AdaptiveRandomForest(BaseOnline):
    def __init__(
        self,
        n_models=10,
        max_features="sqrt",
        lambda_value=6,
        metric=Accuracy(),
        disable_weighted_vote=False,
        drift_detector=ADWIN(),
        warning_detector=ADWIN(),
        grace_period=50,
        max_depth=None,
        split_criterion="info_gain",
        split_confidence=0.01,
        tie_threshold=0.05,
        leaf_prediction="nba",
        nb_threshold=0,
        nominal_attributes=None,
        splitter=None,
        binary_split=False,
        max_size=32,
        memory_estimate_period=2000000,
        stop_mem_management=False,
        remove_poor_attrs=False,
        merit_preprune=True,
        seed=None
    ):
        super().__init__(
            AdaptiveRandomForestClassifier(
                n_models=n_models,
                max_features=max_features,
                lambda_value=lambda_value,
                metric=metric,
                disable_weighted_vote=disable_weighted_vote,
                drift_detector=drift_detector,
                warning_detector=warning_detector,
                grace_period=grace_period,
                max_depth=max_depth,
                split_criterion=split_criterion,
                split_confidence=split_confidence,
                tie_threshold=tie_threshold,
                leaf_prediction=leaf_prediction,
                nb_threshold=nb_threshold,
                nominal_attributes=nominal_attributes,
                splitter=splitter,
                binary_split=binary_split,
                max_size=max_size,
                memory_estimate_period=memory_estimate_period,
                stop_mem_management=stop_mem_management,
                remove_poor_attrs=remove_poor_attrs,
                merit_preprune=merit_preprune,
                seed=seed
            )
        )
