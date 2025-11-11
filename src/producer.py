from kafka import KafkaProducer
import json
from search import random_search


def produce_configs(bootstrap_server="localhost:9092", n_trials=10):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    search_space = {
        "model": ("choice", ["knn", "logreg", "dtree"]),
        "n_neighbors": ("int_range", 1, 10, 2),
        "C": ("uniform", 0.1, 2.0),
        "max_depth": ("int_range", 2, 8, 2),
    }

    for config in random_search(search_space, n_trials=n_trials):
        model = config.get("model")
        if not model:
            print(f"⚠️ Skipping config without model key: {config}")
            continue
        topic = f"hyperparams_{model}"
        print(f"Sending config to {topic}: {config}")
        producer.send(topic, value=config)
    producer.flush()
    print("✅ Dispatched configs for all requested trials.")


if __name__ == "__main__":
    produce_configs()
