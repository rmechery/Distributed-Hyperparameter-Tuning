"""
Auto-detects results_* topics, reads up to N messages each,
plots loss (1-accuracy) per experiment, and saves a PNG.
"""

import uuid, json
import matplotlib
matplotlib.use("Agg")   # safe in headless terminals
import matplotlib.pyplot as plt
from kafka import KafkaConsumer

def collect_all(bootstrap="localhost:9092", limit=20):
    probe = KafkaConsumer(bootstrap_servers=bootstrap)
    topics = probe.topics()
    probe.close()

    res_topics = sorted(t for t in topics if t.startswith("results_"))
    if not res_topics:
        print("⚠️ No 'results_*' topics found. Run main + consumer first.")
        return {}

    print(f"📥 Collecting from: {res_topics}")
    out = {}
    for t in res_topics:
        consumer = KafkaConsumer(
            t,
            bootstrap_servers=bootstrap,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id=f"eval-{t}-{uuid.uuid4().hex[:6]}",  # fresh read from earliest
            auto_offset_reset="earliest"
        )
        trials = []
        for i, msg in enumerate(consumer):
            trials.append(msg.value)
            if i + 1 >= limit: break
        consumer.close()
        out[t.replace("results_", "")] = trials
    return out

def plot_loss_curves(all_results):
    if not all_results:
        print("⚠️ No results to plot.")
        return

    plt.figure(figsize=(8,5))
    for exp, trials in all_results.items():
        if not trials: continue
        scores = [float(t.get("score", 0.0)) for t in trials]
        losses = [1 - s for s in scores]
        plt.plot(range(1, len(losses)+1), losses, marker="o", label=exp)
        print(f"{exp}: {len(trials)} trials, best acc={max(scores):.3f}")

    plt.xlabel("Trial")
    plt.ylabel("Loss (1 - accuracy)")
    plt.title("Loss Minimization Across Experiments")
    plt.legend(title="Experiment")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("loss_curves.png")
    print("✅ Saved plot → loss_curves.png")

if __name__ == "__main__":
    results = collect_all(limit=20)
    plot_loss_curves(results)
