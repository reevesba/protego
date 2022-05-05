""" Typing declarations for models
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "May 5, 2022"
__license__ = "MIT"

from typing import Union, TypeVar

# base
BaseT = TypeVar("BaseT")
BaseBatchT = TypeVar("BaseBatch")
BaseOnlineT = TypeVar("BaseOnline")

# batch
TreeBatchT = TypeVar("DecisionTree")
KNNBatchT = TypeVar("KNeighbors")
LogRegBatchT = TypeVar("LogisticRegressionClassifier")
NaiveBayesBatchT = TypeVar("NaiveBayes")
PerceptronBatchT = TypeVar("PerceptronClassifier")
RandomForestBatchT = TypeVar("RandomForest")

# online
TreeOnlineT = TypeVar("HoeffdingAdaptiveTree")
KNNOnlineT = TypeVar("KNNADWIN")
LogRegOnlineT = TypeVar("LRRiver")
NaiveBayesOnlineT = TypeVar("NBRiver")
PerceptronOnlineT = TypeVar("PRiver")
RandomForestOnlineT = TypeVar("AdaptiveRandomForest")

# online misc
SplitterT = TypeVar("Splitter")
OptimizerT = TypeVar("Optimizer")
BinaryLossT = TypeVar("BinaryLoss")
InterceptLRT = Union[float, TypeVar("Scheduler")]
InitializerT = TypeVar("Initializer")
MultiClassMetricT = TypeVar("MultiClassMetric")
DriftDetectorT = TypeVar("DriftDetector")

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
