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
from protego.models import types as t
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.neighbors import KNNADWINClassifier
from river.linear_model import LogisticRegression
from river.linear_model import Perceptron
from river.naive_bayes import MultinomialNB
from river.ensemble import AdaptiveRandomForestClassifier
from river.metrics import Accuracy
from river.drift import ADWIN
from typing import Union


class HoeffdingAdaptiveTree(BaseOnline):
    def __init__(
        self: t.TreeOnlineT,
        grace_period: int = 200,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        split_confidence: float = 1e-07,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: t.SplitterT = None,
        bootstrap_sampling: bool = True,
        drift_window_threshold: int = 300,
        adwin_confidence: float = 0.002,
        binary_split: bool = False,
        max_size: int = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int = None
    ) -> None:
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
        self: t.KNNOnlineT,
        n_neighbors: int = 5,
        window_size: int = 1000,
        leaf_size: int = 30,
        p: int = 2
    ) -> None:
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
        self: t.LogRegOnlineT,
        optimizer: t.OptimizerT = None,
        loss: t.BinaryLossT = None,
        l2: float = 0.0,
        intercept_init: float = 0.0,
        intercept_lr: t.InterceptLRT = 0.01,
        clip_gradient: float = 1000000000000.0,
        initializer: t.InitializerT = None
    ) -> None:
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
        self: t.NaiveBayesOnlineT,
        alpha: float = 1.0
    ) -> None:
        super().__init__(
            MultinomialNB(
                alpha=alpha
            )
        )


class PerceptronClassifier(BaseOnline):
    def __init__(
        self: t.PerceptronOnlineT,
        l2: float = 0.0,
        clip_gradient: float = 1000000000000.0,
        initializer: t.InitializerT = None
    ) -> None:
        super().__init__(
            Perceptron(
                l2=l2,
                clip_gradient=clip_gradient,
                initializer=initializer
            )
        )


class AdaptiveRandomForest(BaseOnline):
    def __init__(
        self: t.RandomForestOnlineT,
        n_models: int = 10,
        max_features: Union[bool, str, int] = "sqrt",
        lambda_value: int = 6,
        metric: t.MultiClassMetricT = Accuracy(),
        disable_weighted_vote: bool = False,
        drift_detector: t.DriftDetectorT = ADWIN(),
        warning_detector: t.DriftDetectorT = ADWIN(),
        grace_period: int = 50,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: t.SplitterT = None,
        binary_split: bool = False,
        max_size: int = 32,
        memory_estimate_period: int = 2000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int = None
    ) -> None:
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
