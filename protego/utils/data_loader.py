""" Load Delivered Dataset
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "April 3, 2022"
__license__ = "None"

import os
import pandas as pd


class DataLoader:
    def __init__(self):
        self.dir, _ = os.path.split(os.path.split(__file__)[0])
        self.filepath_a = self.dir + '/datasets/delivered/sqli_train.csv'
        self.filepath_b = self.dir + '/datasets/delivered/sqli_test.csv'

    def load(self):
        df_a = pd.read_csv(self.filepath_a)
        df_b = pd.read_csv(self.filepath_b)
        return pd.concat([df_a, df_b])
