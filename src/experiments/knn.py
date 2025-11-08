from typing import Dict
from search import random_search

TOPIC_IN  = "hyperparams_knn"
TOPIC_OUT = "results_knn"

def search_space() -> Dict:
    return {
        "model": ("choice", ["knn"]),
        "n_neighbors": ("int_range", 1, 16, 2),
    }

def configs(n_trials: int):
    yield from random_search(search_space(), n_trials)
