import random
from typing import Dict, Any, Iterator, Tuple

def _sample_value(rule: Tuple) -> Any:
    kind = rule[0]
    if kind == "choice":
        return random.choice(rule[1])
    if kind == "uniform":
        return random.uniform(rule[1], rule[2])
    if kind == "int_range":
        start, stop = rule[1], rule[2]
        step = rule[3] if len(rule) > 3 else 1
        return random.randrange(start, stop, step)
    raise ValueError(f"Unknown search rule: {rule}")

def random_search(space: Dict[str, Any], n_trials: int, unique: bool = False) -> Iterator[Dict[str, Any]]:
    """Yield random hyperparameter combinations.

    If `unique` is True, configurations are deduplicated (best-effort). For discrete spaces,
    this prevents recomputing the same hyperparameters across workers.
    """
    seen = set()
    generated = 0
    max_attempts = max(n_trials * 10, 100)

    attempts = 0
    while generated < n_trials and attempts < max_attempts:
        params = {name: _sample_value(rule) for name, rule in space.items()}
        if unique:
            key = tuple(sorted(params.items()))
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
        yield params
        generated += 1
        attempts = 0
    if unique and generated < n_trials:
        print(f"⚠️ random_search could only generate {generated}/{n_trials} unique configs.")
