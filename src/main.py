import argparse
import importlib
import json
import os
import sys
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic, NewPartitions
from kafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS = ("knn", "logreg", "dtree")
DATASETS = ("iris", "wine", "digits", "breast_cancer")

def _ensure_topic(admin: KafkaAdminClient, topic: str, partitions: int):
    try:
        desc = admin.describe_topics([topic])[0]
        current = len(desc.get("partitions", []))
        if current < partitions:
            admin.create_partitions({topic: NewPartitions(total_count=partitions)})
            print(f"ℹ️ Increased topic '{topic}' partitions {current} → {partitions}")
        return
    except UnknownTopicOrPartitionError:
        pass

    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=1)])
        print(f"ℹ️ Created topic '{topic}' with {partitions} partitions.")
    except TopicAlreadyExistsError:
        pass

def run_experiment(exp_name: str, n_trials: int, dataset: str, bootstrap: str, partitions: int):
    mod = importlib.import_module(f"experiments.{exp_name}")
    topic = getattr(mod, "TOPIC_IN")
    out_topic = getattr(mod, "TOPIC_OUT", f"results_{exp_name}")

    admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="experiment-admin")
    try:
        _ensure_topic(admin, topic, partitions)
        _ensure_topic(admin, out_topic, partitions)
    finally:
        admin.close()

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print(f"▶ Running experiment '{exp_name}' on dataset '{dataset}' → topic='{topic}', trials={n_trials}, partitions={partitions}")

    for cfg in mod.configs(n_trials):
        payload = dict(cfg)
        payload["dataset"] = dataset
        print(f"→ Sending: {payload}")
        producer.send(topic, value=payload)
    producer.flush()
    print("✅ Done.")

def parse_args():
    parser = argparse.ArgumentParser(description="Publish hyperparameter configs to Kafka.")
    parser.add_argument("experiment", choices=EXPERIMENTS, help="Experiment module to run.")
    parser.add_argument("-t", "--trials", type=int, default=12, help="Number of configs to send.")
    parser.add_argument("--dataset", choices=DATASETS, default="iris", help="Dataset to train on.")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap servers.")
    parser.add_argument("--partitions", type=int, default=1, help="Ensure topics have at least this many partitions.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.experiment, args.trials, args.dataset, args.bootstrap, max(1, args.partitions))
