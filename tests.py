""" Used for TDD
"""

__author__ = "Bradley Reeves"
__email__ = "reevesbra@outlook.com"
__date__ = "March 27, 2022"
__license__ = "MIT"

import unittest
from protego.models.batch import (
    DecisionTree, KNeighbors, LogisticRegressionClassifier,
    NaiveBayes, PerceptronClassifier, RandomForest
)
from protego.models.online import (
    HoeffdingAdaptiveTree, KNNADWIN, LogisticRegressionClassifier as OnlineLR,
    NaiveBayes as OnlineNB, PerceptronClassifier as OnlinePer,
    AdaptiveRandomForest
)
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
        split = int(len(dataset)*0.8)
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
        train_set = train_data[features]
        train_lbl = train_data["label"]

        test_set = test_data[features]
        test_lbl = test_data["label"]

        # Batch Tests
        treb = DecisionTree()
        knnb = KNeighbors()
        logb = LogisticRegressionClassifier()
        mnbb = NaiveBayes()
        perb = PerceptronClassifier()
        ranb = RandomForest()

        self.assertIsInstance(
            treb,
            DecisionTree,
            "Object isn't instance of DecisionTree"
        )
        self.assertIsInstance(
            knnb,
            KNeighbors,
            "Object isn't instance of KNeighbors"
        )
        self.assertIsInstance(
            logb,
            LogisticRegressionClassifier,
            "Object isn't instance of LogisticRegressionClassifier"
        )
        self.assertIsInstance(
            mnbb,
            NaiveBayes,
            "Object isn't instance of NaiveBayes"
        )
        self.assertIsInstance(
            perb,
            PerceptronClassifier,
            "Object isn't instance of PerceptronClassifier"
        )
        self.assertIsInstance(
            ranb,
            RandomForest,
            "Object isn't instance of RandomForest"
        )
        treb.train(train_set, train_lbl)
        knnb.train(train_set, train_lbl)
        logb.train(train_set, train_lbl)
        mnbb.train(train_set, train_lbl)
        perb.train(train_set, train_lbl)
        ranb.train(train_set, train_lbl)

        treb.save("decision_tree_treb.pkl")
        knnb.save("decision_tree_knnb.pkl")
        logb.save("decision_tree_logb.pkl")
        mnbb.save("decision_tree_mnbb.pkl")
        perb.save("decision_tree_perb.pkl")
        ranb.save("decision_tree_ranb.pkl")

        treb.load("decision_tree_treb.pkl")
        knnb.load("decision_tree_knnb.pkl")
        logb.load("decision_tree_logb.pkl")
        mnbb.load("decision_tree_mnbb.pkl")
        perb.load("decision_tree_perb.pkl")
        ranb.load("decision_tree_ranb.pkl")

        y_pred_treb = treb.predict(test_set)
        y_pred_knnb = knnb.predict(test_set)
        y_pred_logb = logb.predict(test_set)
        y_pred_mnbb = mnbb.predict(test_set)
        y_pred_perb = perb.predict(test_set)
        y_pred_ranb = ranb.predict(test_set)

        print("-- Batch Accuracies --")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_treb):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_knnb):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_logb):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_mnbb):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_perb):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_ranb):3.3f}%")

        # Online Tests
        treo = HoeffdingAdaptiveTree()
        knno = KNNADWIN()
        logo = OnlineLR()
        mnbo = OnlineNB()
        pero = OnlinePer()
        rano = AdaptiveRandomForest()

        self.assertIsInstance(
            treo,
            HoeffdingAdaptiveTree,
            "Object isn't instance of HoeffdingAdaptiveTree"
        )
        self.assertIsInstance(
            knno,
            KNNADWIN,
            "Object isn't instance of KNNADWIN"
        )
        self.assertIsInstance(
            logo,
            OnlineLR,
            "Object isn't instance of OnlineLR"
        )
        self.assertIsInstance(
            mnbo,
            OnlineNB,
            "Object isn't instance of OnlineNB"
        )
        self.assertIsInstance(
            pero,
            OnlinePer,
            "Object isn't instance of OnlinePer"
        )
        self.assertIsInstance(
            rano,
            AdaptiveRandomForest,
            "Object isn't instance of AdaptiveRandomForest"
        )

        treo.train(train_set, train_lbl)
        knno.train(train_set, train_lbl)
        logo.train(train_set, train_lbl)
        mnbo.train(train_set, train_lbl)
        pero.train(train_set, train_lbl)
        rano.train(train_set, train_lbl)

        treo.save("decision_tree_treo.pkl")
        knno.save("decision_tree_knno.pkl")
        logo.save("decision_tree_logo.pkl")
        mnbo.save("decision_tree_mnbo.pkl")
        pero.save("decision_tree_pero.pkl")
        rano.save("decision_tree_rano.pkl")

        treo.load("decision_tree_treo.pkl")
        knno.load("decision_tree_knno.pkl")
        logo.load("decision_tree_logo.pkl")
        mnbo.load("decision_tree_mnbo.pkl")
        pero.load("decision_tree_pero.pkl")
        rano.load("decision_tree_rano.pkl")

        y_pred_treo = treo.predict(test_set)
        y_pred_knno = knno.predict(test_set)
        y_pred_logo = logo.predict(test_set)
        y_pred_mnbo = mnbo.predict(test_set)
        y_pred_pero = pero.predict(test_set)
        y_pred_rano = rano.predict(test_set)

        print("-- Online Accuracies --")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_treo):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_knno):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_logo):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_mnbo):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_pero):3.3f}%")
        print(f"Accuracy: {100*accuracy_score(test_lbl, y_pred_rano):3.3f}%")

        print("All tests passed.")


if __name__ == "__main__":
    unittest.main()
