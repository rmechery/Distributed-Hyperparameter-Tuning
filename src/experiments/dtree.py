from typing import Dict
from search import random_search

TOPIC_IN  = "hyperparams_dtree"
TOPIC_OUT = "results_dtree"

def search_space() -> Dict:
    return {
        "model": ("choice", ["dtree"]),
        "max_depth": ("int_range", 2, 30, 1),
    }

def configs(n_trials: int):
    yield from random_search(search_space(), n_trials, unique=True)
