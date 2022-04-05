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
__license__ = "None"

from protego.models.base import BaseBatch
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class DecisionTree(BaseBatch):
    def __init__(
        self,
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0
    ):
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
        self,
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
    ):
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
        self,
        penalty='l2',
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=100,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None
    ):
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
        alpha=1.0,
        fit_prior=True,
        class_prior=None
    ):
        super().__init__(
            MultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior,
                class_prior=class_prior
            )
        )


class PerceptronClassifier(BaseBatch):
    def __init__(
        self,
        penalty=None,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        eta0=1,
        n_jobs=None,
        random_state=None,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False
    ):
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
        self,
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None
    ):
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
