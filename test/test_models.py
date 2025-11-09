import unittest
import sys
import os

from src.models import train_model as tmodel

class TestTrainModel(unittest.TestCase):
    def testlogreg(self):
        result = tmodel("logreg", {"C": 0.5, "max_iter": 1000})
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def testknn(self):
        result = tmodel("knn", {"n_neighbors": 3})
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def testdectree(self):
        result = tmodel("dtree", {"max_depth": 4})
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

if __name__ == "__main__":
    unittest.main()
