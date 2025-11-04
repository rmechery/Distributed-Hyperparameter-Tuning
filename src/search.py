import random
from typing import Dict, Any, Callable, List

def sample_params(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a single hyperparameter configuration."""
    sampled = {}
    for name, rule in search_space.items():
        kind = rule[0]
        if kind == "choice":
            sampled[name] = random.choice(rule[1])
        elif kind == "uniform":
            sampled[name] = random.uniform(rule[1], rule[2])
        elif kind == "int_range":
            start, stop, step = rule[1], rule[2], rule[3]
            sampled[name] = random.randrange(start, stop, step)
    return sampled


def random_search(
    search_space: Dict[str, Any],
    eval_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 10
):
    """Perform basic random search for best hyperparameters."""
    results = []
    for _ in range(n_trials):
        params = sample_params(search_space)
        score = eval_fn(params)
        results.append((params, score))
    return sorted(results, key=lambda x: x[1], reverse=True)
