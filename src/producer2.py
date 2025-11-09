from kafka import KafkaProducer
import json
from search import random_search

MODEL_LIST = ["knn", "logreg", "dtree"]

def produce_configs(target="all", n_trials=10, bootstrap_server="localhost:9092"):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    search_space = {
        "model": ("choice", MODEL_LIST),
        "n_neighbors": ("int_range", 1, 10, 2),
        "C": ("uniform", 0.1, 2.0),
        "max_depth": ("int_range", 2, 8, 2),
    }

    if target != "all" and target not in MODEL_LIST:
        print(f"Unknown target '{target}', must be one of {MODEL_LIST} or 'all'")
        return

    for config in random_search(search_space, n_trials=n_trials):
        if target != "all":
            config["model"] = target  # override model in the payload
        model = config["model"]
        topic = f"hyperparams_{model}"
        print(f"Sending to {topic}: {config}")
        producer.send(topic, value=config)
        producer.flush()


    print("✅ Dispatched configs for all requested trials.")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    produce_configs(target, n_trials)
