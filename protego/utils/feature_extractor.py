""" Feature extraction/selection
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "November 30, 2021"
__license__ = "None"

import re
import pandas as pd


class FeatureExtractor:
    def transform(self, data, include_payload):
        return self.__get_features(data, include_payload)

    def __get_num_chars(self, payload):
        return len(payload)

    def __get_num_words(self, payload):
        return len(payload.split())

    def __get_num_special_chars(self, payload):
        return len(re.sub('[\w]+', '', payload))

    def __get_num_ticks(self, payload):
        return payload.count('\'')

    def __get_num_dashes(self, payload):
        return payload.count('-')

    def __get_num_commas(self, payload):
        return payload.count(',')

    def __get_num_pipes(self, payload):
        return payload.count('|')

    def __get_num_equals(self, payload):
        return payload.count('=')

    def __get_num_keywords(self, payload):
        keywords = ['select', 'from', 'where', 'union',
                    'sleep', 'or', 'and', 'like', 'order']
        count = 0

        word_list = payload.split()
        for word in word_list:
            if word.lower() in keywords:
                count += 1

        return count

    def __get_num_parens(self, payload):
        return payload.count('(') + payload.count(')')

    def __get_avg_word_len(self, payload):
        words = payload.split()
        if len(words) > 0:
            return sum(len(word) for word in words)/len(words)
        else:
            return len(payload)

    def __get_num_white_spaces(self, payload):
        return payload.count(' ')

    def __get_num_comments(self, payload):
        return payload.count('--')

    def __get_features(self, data, include_payload=False):
        rows = []
        for i, row in data.iterrows():
            cols = []
            payload = str(row['payload'])
            label = row['label']

            cols.append(self.__get_num_chars(payload))
            cols.append(self.__get_num_words(payload))
            cols.append(self.__get_num_special_chars(payload))
            cols.append(self.__get_num_ticks(payload))
            cols.append(self.__get_num_dashes(payload))
            cols.append(self.__get_num_commas(payload))
            cols.append(self.__get_num_pipes(payload))
            cols.append(self.__get_num_equals(payload))
            cols.append(self.__get_num_keywords(payload))
            cols.append(self.__get_num_parens(payload))
            cols.append(self.__get_avg_word_len(payload))
            cols.append(self.__get_num_white_spaces(payload))
            cols.append(self.__get_num_comments(payload))
            cols.append(label)
            if include_payload:
                cols.append(payload)
            rows.append(cols)

        df = pd.DataFrame(rows,
                          columns=['num_chars', 'num_words',
                                   'num_special_chars', 'num_ticks',
                                   'num_dashes', 'num_commas', 'num_pipes',
                                   'num_equals', 'num_keywords', 'num_parens',
                                   'avg_word_len', 'num_white_spaces',
                                   'num_comments', 'label'])

        return df
