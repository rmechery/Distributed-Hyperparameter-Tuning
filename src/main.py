import importlib, sys, os, json
from kafka import KafkaProducer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_experiment(exp_name: str, n_trials: int = 12, bootstrap="localhost:9092"):
    mod = importlib.import_module(f"experiments.{exp_name}")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    topic = getattr(mod, "TOPIC_IN")
    print(f"▶ Running experiment '{exp_name}' → topic='{topic}', trials={n_trials}")

    for cfg in mod.configs(n_trials):
        print(f"→ Sending: {cfg}")
        producer.send(topic, value=cfg)
    producer.flush()
    print("✅ Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [knn|logreg|dtree] [trials?]")
        sys.exit(1)
    exp = sys.argv[1].lower()
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    run_experiment(exp, trials)
