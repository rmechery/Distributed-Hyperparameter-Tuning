from typing import Dict
from search import random_search

TOPIC_IN  = "hyperparams_logreg"
TOPIC_OUT = "results_logreg"

def search_space() -> Dict:
    return {
        "model": ("choice", ["logreg"]),
        "C": ("uniform", 1e-4, 1e3),
    }

def configs(n_trials: int):
    yield from random_search(search_space(), n_trials)
