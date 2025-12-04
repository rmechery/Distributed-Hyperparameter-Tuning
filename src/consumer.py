import json
from kafka import KafkaConsumer, KafkaProducer
from src.models import train_model

def consume_and_train(bootstrap="localhost:9092"):
    # Probe topics
    probe = KafkaConsumer(bootstrap_servers=bootstrap)
    topics = probe.topics()
    probe.close()

    in_topics = sorted(t for t in topics if t.startswith("hyperparams_"))
    if not in_topics:
        print("⚠️ No 'hyperparams_*' topics found. Run src/main.py first.")
        return

    print(f"👂 Worker subscribing to: {in_topics}")

    consumer = KafkaConsumer(
        *in_topics,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="worker-group",          # same group → load-balanced across workers
        auto_offset_reset="earliest"
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    for msg in consumer:
        in_topic = msg.topic
        exp_name = in_topic.replace("hyperparams_", "")
        params = msg.value
        model = params.get("model", exp_name)

        try:
            score = train_model(model, params)
            out_topic = f"results_{exp_name}"
            payload = {"params": params, "score": score, "model": model}
            print(f"[{exp_name}] {params} → {score:.3f}")
            producer.send(out_topic, value=payload)
            producer.flush()
        except Exception as e:
            print(f"⚠️ Error for {exp_name}: {e} | params={params}")

if __name__ == "__main__":
    consume_and_train()
