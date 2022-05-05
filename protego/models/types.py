""" Typing declarations for models
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "May 5, 2022"
__license__ = "MIT"

from typing import Union
from protego.models.base import (
    Base, BaseBatch, BaseOnline
)
from protego.models.batch import (
    DecisionTree, KNeighbors,  LogisticRegressionClassifier,
    NaiveBayes, PerceptronClassifier, RandomForest
)
from protego.models.online import (
    HoeffdingAdaptiveTree, KNNADWIN, LogisticRegressionClassifier as LRRiver,
    NaiveBayes as NBRiver, PerceptronClassifier as PRiver, AdaptiveRandomForest
)
from river.tree.splitter.base import Splitter
from river.optim import Optimizer
from river.optim.losses import BinaryLoss
from river.optim.initializers import Initializer
from river.schedulers import Scheduler
from river.metrics.base import MultiClassMetric
from river.base import DriftDetector

# base
BaseT = type[Base]
BaseBatchT = type[BaseBatch]
BaseOnlineT = type[BaseOnline]

# batch
TreeBatchT = type[DecisionTree]
KNNBatchT = type[KNeighbors]
LogRegBatchT = type[LogisticRegressionClassifier]
NaiveBayesBatchT = type[NaiveBayes]
PerceptronBatchT = type[PerceptronClassifier]
RandomForestBatchT = type[RandomForest]

# online
TreeOnlineT = type[HoeffdingAdaptiveTree]
KNNOnlineT = type[KNNADWIN]
LogRegOnlineT = type[LRRiver]
NaiveBayesOnlineT = type[NBRiver]
PerceptronOnlineT = type[PRiver]
RandomForestOnlineT = type[AdaptiveRandomForest]

# online misc
SplitterT = type[Splitter]
OptimizerT = type[Optimizer]
BinaryLossT = type[BinaryLoss]
InterceptLRT = Union[float, type[Scheduler]]
InitializerT = type[Initializer]
MultiClassMetricT = type[MultiClassMetric]
DriftDetectorT = type[DriftDetector]

BatchModelT = Union[
    TreeBatchT, KNNBatchT, LogRegBatchT,
    NaiveBayesBatchT, PerceptronBatchT, RandomForestBatchT
]

OnlineModelT = Union[
    TreeOnlineT, KNNOnlineT, LogRegOnlineT,
    NaiveBayesOnlineT, PerceptronOnlineT, RandomForestOnlineT
]

ModelT = Union[BatchModelT, OnlineModelT]

NumberT = Union[int, float]
MaxFeaturesT = Union[NumberT, str]
ClassWeightT = Union[dict, list, str]
ClassPriorT = Union[tuple, list]
