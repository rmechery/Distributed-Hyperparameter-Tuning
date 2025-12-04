import pytest, sys, os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.search import random_search

#cases of multiple hyperparameters
@patch("random.choice", side_effect=[32, 16])
@patch("random.uniform", side_effect=[0.003, 0.007])
@patch("random.randrange", side_effect=[1, 3])
def test_random_search(mock_randrange, mock_uniform, mock_choice):
    results = list(random_search({"lr": ("uniform", 0.001, 0.01), "batch_size": ("choice", [16, 32, 64]), "layers": ("int_range", 1, 5, 1)}, 2))
    assert len(results) == 2 # check if output is two dictionaries
    assert results[0] == {"lr": 0.003, "batch_size": 32, "layers": 1} and results[1] == {"lr": 0.007, "batch_size": 16, "layers": 3}

    for r in results:
        assert set(r.keys()) == {"lr", "batch_size", "layers"} # Check keys

#case of one hyperparameter
@patch("random.choice", side_effect=[2, 3])
@patch("random.uniform", side_effect=[3, 0.007])
@patch("random.randrange", side_effect=[0, 3])
def test_random_search2(mock_randrange, mock_uniform, mock_choice):
    results = list(random_search({"batch_size": ("choice", [2])}, 1))
    assert len(results) == 1 and results[0] == {"batch_size": 2} 
    assert set(results[0].keys()) == {"batch_size"}