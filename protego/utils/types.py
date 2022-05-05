""" Typing declarations for utils
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "May 5, 2022"
__license__ = "MIT"

from protego.utils.data_loader import DataLoader
from protego.utils.feature_extractor import FeatureExtractor

DataLoaderT = type[DataLoader]
FeatureExtT = type[FeatureExtractor]
