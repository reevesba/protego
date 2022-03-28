""" Naive Bayes
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "None"

from base import Base
from river.naive_bayes import MultinomialNB


class NaiveBayes(Base):
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha)
        super().__init__(self.model)
