""" Perceptron
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.linear_model import Perceptron


class PerceptronClassifier(Base):
    def __init__(self, l2=0.0, clip_gradient=1000000000000.0,
                 initializer=None):
        self.model = Perceptron(
            l2, clip_gradient, initializer
        )
        super().__init__(self.model)
