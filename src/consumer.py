import json
import os
import socket
import time
from kafka import KafkaConsumer, KafkaProducer
from src.models import train_model

def consume_and_train(bootstrap="localhost:9092"):
    host = socket.gethostname().split(".")[0]
    worker_id = f"{host}-{os.getpid()}"
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="worker-group",          # same group → load-balanced across workers
        auto_offset_reset="earliest"
    )
    consumer.subscribe(pattern="^hyperparams_")

    existing = sorted(t for t in consumer.topics() if t.startswith("hyperparams_"))
    if existing:
        print(f"👂 Worker subscribed to: {existing}")
    else:
        print("👂 Worker waiting for topics matching 'hyperparams_*' (will attach when they appear)")

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    for msg in consumer:
        in_topic = msg.topic
        exp_name = in_topic.replace("hyperparams_", "")
        params = msg.value
        model = params.get("model", exp_name)
        dataset = params.get("dataset", "iris")

        try:
            start = time.perf_counter()
            score = train_model(model, params, dataset)
            duration = time.perf_counter() - start
            out_topic = f"results_{exp_name}"
            payload = {
                "params": params,
                "score": score,
                "model": model,
                "worker_id": worker_id,
                "duration_sec": duration,
            }
            print(f"[{exp_name}] {params} → {score:.3f} ({duration:.2f}s, worker={worker_id})")
            producer.send(out_topic, value=payload)
            producer.flush()
        except Exception as e:
            print(f"⚠️ Error for {exp_name}: {e} | params={params}")

if __name__ == "__main__":
    consume_and_train()
