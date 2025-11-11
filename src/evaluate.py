"""
Auto-detects results_* topics, reads up to N messages each,
plots loss (1-accuracy) per experiment, and saves a PNG.
"""

import uuid, json, time, os
import matplotlib
matplotlib.use("Agg")   # safe in headless terminals
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
from collections import defaultdict
from datetime import datetime

def collect_all(bootstrap="localhost:9092", limit=20, wait=True, idle_timeout=5.0, from_latest=True):
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id=f"eval-{uuid.uuid4().hex[:6]}",
        auto_offset_reset="latest" if from_latest else "earliest"
    )
    consumer.subscribe(pattern="^results_")
    print("📥 Subscribed to 'results_*'. Waiting for data (Ctrl+C to stop)...")
    if from_latest:
        print("ℹ️ Reading only messages produced after evaluator start (auto_offset=latest).")

    results = defaultdict(list)
    last_update = None
    last_notice = time.time()

    try:
        while True:
            records = consumer.poll(timeout_ms=1000)
            if not records:
                now = time.time()
                if results and last_update and (now - last_update) >= idle_timeout:
                    break
                if not results:
                    if now - last_notice >= idle_timeout:
                        print("⏳ Waiting for first results...")
                        last_notice = now
                    continue
                if wait:
                    continue
                break

            last_update = time.time()
            for tp, messages in records.items():
                exp = tp.topic.replace("results_", "")
                bucket = results[exp]
                for msg in messages:
                    if len(bucket) >= limit:
                        break
                    bucket.append(msg.value)
                print(f"➕ Collected {len(messages)} result(s) for '{exp}' (total {len(bucket)})")

            if not wait and results:
                break
    finally:
        consumer.close()

    if not results:
        print("⚠️ No results were received before exit.")
    return dict(results)

def plot_loss_curves(all_results):
    if not all_results:
        print("⚠️ No results to plot.")
        return

    plt.figure(figsize=(8,5))
    plotted_experiments = []
    for exp, trials in all_results.items():
        if not trials:  # collect_all already filters empty, still guard
            continue
        scores = [float(t.get("score", 0.0)) for t in trials]
        losses = [1 - s for s in scores]
        plt.plot(range(1, len(losses)+1), losses, marker="o", label=exp)
        print(f"{exp}: {len(trials)} trials, best acc={max(scores):.3f}")
        plotted_experiments.append(exp)

    if not plotted_experiments:
        print("⚠️ Collected topics but none contained trials. Skipping plot.")
        return

    exp_slug = "-".join(sorted(plotted_experiments)) or "results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("plots", exist_ok=True)
    outfile = os.path.join("plots", f"loss_{exp_slug}_{timestamp}.png")

    plt.xlabel("Trial")
    plt.ylabel("Loss (1 - accuracy)")
    plt.title("Loss Minimization Across Experiments")
    plt.legend(title="Experiment")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")

if __name__ == "__main__":
    results = collect_all(limit=20)
    plot_loss_curves(results)
