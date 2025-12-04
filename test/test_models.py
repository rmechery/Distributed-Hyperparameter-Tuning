import unittest, sys, os, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    def test_unknown_model(self):
        with self.assertRaises(ValueError) as context:
            tmodel("svm", {})
        self.assertIn("Unknown model", str(context.exception))    

    
if __name__ == "__main__":
    unittest.main()
