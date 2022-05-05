""" Batch Models

    1. Decision Tree
    2. K-Nearest Neighbors
    3. Logistic Regression
    4. Naive Bayes
    5. Perceptron
    6. Random Forest
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "April 3, 2022"
__license__ = "MIT"

from protego.models.base import BaseBatch
from protego.models import types as t
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class DecisionTree(BaseBatch):
    def __init__(
        self: t.TreeBatchT,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: int = None,
        min_samples_split: t.NumberT = 2,
        min_samples_leaf: t.NumberT = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: t.MaxFeaturesT = None,
        random_state: int = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        class_weight: t.ClassWeightT = None,
        ccp_alpha: float = 0.0
    ) -> None:
        super().__init__(
            DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                class_weight=class_weight,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha
            )
        )


class KNeighbors(BaseBatch):
    def __init__(
        self: t.KNNBatchT,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: dict = None,
        n_jobs: int = None
    ) -> None:
        super().__init__(
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs
            )
        )


class LogisticRegressionClassifier(BaseBatch):
    def __init__(
        self: t.LogRegBatchT,
        penalty: str = "l2",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: t.ClassWeightT = None,
        random_state: int = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int = None,
        l1_ratio: float = None
    ) -> None:
        super().__init__(
            LogisticRegression(
                penalty=penalty,
                dual=dual,
                tol=tol,
                C=C,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                random_state=random_state,
                solver=solver,
                max_iter=max_iter,
                multi_class=multi_class,
                verbose=verbose,
                warm_start=warm_start,
                n_jobs=n_jobs,
                l1_ratio=l1_ratio
            )
        )


class NaiveBayes(BaseBatch):
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: t.ClassPriorT = None
    ) -> None:
        super().__init__(
            MultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior,
                class_prior=class_prior
            )
        )


class PerceptronClassifier(BaseBatch):
    def __init__(
        self: t.PerceptronBatchT,
        penalty: str = None,
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        verbose: int = 0,
        eta0: float = 1,
        n_jobs: int = None,
        random_state: int = None,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        class_weight: t.ClassWeightT = None,
        warm_start: bool = False
    ) -> None:
        super().__init__(
            Perceptron(
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                shuffle=shuffle,
                verbose=verbose,
                eta0=eta0,
                n_jobs=n_jobs,
                random_state=random_state,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                class_weight=class_weight,
                warm_start=warm_start
            )
        )


class RandomForest(BaseBatch):
    def __init__(
        self: t.RandomForestBatchT,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: t.Number = 2,
        min_samples_leaf: t.Number = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str = "auto",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: t.ClassWeightT = None,
        ccp_alpha: float = 0.0,
        max_samples: t.Number = None
    ) -> None:
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )
        )
