""" Used for TDD
"""

import unittest
from protego.models.batch import DecisionTree
from protego.utils.feature_extractor import FeatureExtractor
from protego.utils.data_loader import DataLoader
from sklearn.metrics import accuracy_score


class TestProtego(unittest.TestCase):
    def test_harness(self):
        data_loader = DataLoader()

        # Ensure object get instantiated
        self.assertIsInstance(
            data_loader,
            DataLoader,
            "Object isn't instance of DataLoader"
        )

        dataset = data_loader.load()

        # Split into train/test sets
        split = int(len(dataset)*0.2)
        train_data = dataset.loc[:split, :]
        test_data = dataset.loc[split + 1:, :]

        # Extract Features
        fe = FeatureExtractor()

        # Ensure object get instantiated
        self.assertIsInstance(
            fe,
            FeatureExtractor,
            "Object isn't instance of FeatureExtractor"
        )

        train_data = fe.transform(train_data)
        train_data.dropna(inplace=True)
        train_data.drop_duplicates(inplace=True)
        train_data.reset_index(inplace=True, drop=True)

        test_data = fe.transform(test_data)
        test_data.dropna(inplace=True)
        test_data.drop_duplicates(inplace=True)
        test_data.reset_index(inplace=True, drop=True)

        # Ensure train/test data exists
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)

        # Process dataset for training/testing
        features = train_data.columns[:-1]
        training_set = train_data[features]
        training_lbl = train_data['label']

        testing_set = test_data[features]
        testing_lbl = test_data['label']

        # Modelling
        model = DecisionTree()

        # Ensure object get instantiated
        self.assertIsInstance(
            model,
            DecisionTree,
            "Object isn't instance of DecisionTree"
        )

        model.train(training_set, training_lbl)

        model.save("decision_tree_1.pkl")
        model.load("decision_tree_1.pkl")

        y_pred = model.predict(testing_set)

        print(f'Accuracy: {accuracy_score(testing_lbl, y_pred)}')
        print('All tests passed.')


if __name__ == "__main__":
    unittest.main()
