import random
from typing import Dict, Any, Iterator

def random_search(space: Dict[str, Any], n_trials: int) -> Iterator[Dict[str, Any]]:
    """Yield random hyperparameter combinations."""
    for _ in range(n_trials):
        params = {}
        for name, rule in space.items():
            kind = rule[0]
            if kind == "choice":
                params[name] = random.choice(rule[1])
            elif kind == "uniform":
                params[name] = random.uniform(rule[1], rule[2])
            elif kind == "int_range":
                params[name] = random.randrange(rule[1], rule[2], rule[3])
        yield params
