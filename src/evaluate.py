"""
Auto-detects results_* topics, reads up to N messages each,
and saves plots + tables for quick experiment comparison.
"""

import argparse
import csv
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from statistics import mean, median, pstdev

import matplotlib
matplotlib.use("Agg")   # safe in headless terminals
import matplotlib.pyplot as plt
from kafka import KafkaConsumer

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
                    payload = dict(msg.value)
                    payload["_partition"] = msg.partition
                    payload["_offset"] = msg.offset
                    payload["_timestamp"] = msg.timestamp  # ms since epoch
                    bucket.append(payload)
                print(f"➕ Collected {len(messages)} result(s) for '{exp}' (total {len(bucket)})")

            if not wait and results:
                break
    finally:
        consumer.close()

    if not results:
        print("⚠️ No results were received before exit.")
    return dict(results)

def plot_loss_curves(all_results, outfile=None):
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

    if not outfile:
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

def plot_accuracy_curves(all_results, outfile):
    if not all_results:
        return False

    plt.figure(figsize=(8,5))
    plotted = []
    for exp, trials in all_results.items():
        if not trials:
            continue
        scores = [float(t.get("score", 0.0)) for t in trials]
        plt.plot(range(1, len(scores)+1), scores, marker="o", label=exp)
        plotted.append(exp)
    if not plotted:
        return False

    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Trial")
    plt.ylim(0, 1.05)
    plt.legend(title="Experiment")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")
    return True

def plot_best_bars(summary_rows, outfile):
    if not summary_rows:
        return False

    labels = [row["experiment"] for row in summary_rows]
    bests = [row["best_score"] for row in summary_rows]

    plt.figure(figsize=(8,4))
    bars = plt.bar(labels, bests, color="#4C78A8")
    plt.ylim(0, 1.05)
    plt.ylabel("Best Accuracy")
    plt.title("Best Accuracy by Experiment")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar, score in zip(bars, bests):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")
    return True

def plot_box(all_results, outfile):
    if not all_results:
        return False

    labels, data = [], []
    for exp, trials in all_results.items():
        if not trials:
            continue
        labels.append(exp)
        data.append([float(t.get("score", 0.0)) for t in trials])
    if not labels:
        return False

    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Accuracy")
    plt.title("Score Distribution")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")
    return True

def plot_knn_scatter(all_results, outfile):
    trials = all_results.get("knn", [])
    if not trials:
        return False

    xs, ys = [], []
    for t in trials:
        nn = t.get("params", {}).get("n_neighbors") if isinstance(t.get("params"), dict) else None
        if nn is None:
            nn = t.get("n_neighbors")  # fallback if params flattened
        score = t.get("score")
        if nn is None or score is None:
            continue
        xs.append(int(nn))
        ys.append(float(score))
    if not xs:
        return False

    plt.figure(figsize=(7,5))
    plt.scatter(xs, ys, alpha=0.8, color="#F58518")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.title("KNN: n_neighbors vs Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")
    return True

def plot_duration_curves(all_results, outfile):
    if not all_results:
        return False

    plt.figure(figsize=(8,5))
    plotted = []
    for exp, trials in all_results.items():
        durations = [t.get("duration_sec") for t in trials if t.get("duration_sec") is not None]
        if not durations:
            continue
        plt.plot(range(1, len(durations)+1), durations, marker="o", label=exp)
        plotted.append(exp)
    if not plotted:
        return False

    plt.xlabel("Trial")
    plt.ylabel("Duration (s)")
    plt.title("Per-trial Duration")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Experiment")
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")
    return True

def plot_worker_time(worker_rows, outfile):
    if not worker_rows:
        return False

    labels = [r["worker_id"] for r in worker_rows]
    avg_durations = [r["avg_duration_sec"] for r in worker_rows]
    tasks = [r["tasks"] for r in worker_rows]

    fig, ax1 = plt.subplots(figsize=(8,5))
    bars = ax1.bar(labels, avg_durations, color="#4C78A8", label="Avg duration (s)")
    ax1.set_ylabel("Avg duration (s)")
    ax1.set_ylim(0, max(avg_durations) * 1.2 if avg_durations else 1)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax1.set_title("Worker Time & Task Count")

    ax2 = ax1.twinx()
    ax2.plot(labels, tasks, color="#F58518", marker="o", label="Tasks")
    ax2.set_ylabel("Tasks")

    # combine legends
    handles, labels_all = [], []
    for h, l in zip(bars, ["Avg duration (s)"]):
        handles.append(h)
        labels_all.append(l)
    line = ax2.lines[0]
    handles.append(line)
    labels_all.append("Tasks")
    ax1.legend(handles, labels_all, loc="upper right")

    plt.tight_layout()
    plt.savefig(outfile)
    print(f"✅ Saved plot → {outfile}")
    return True

def summarize(all_results):
    summary = []
    for exp, trials in all_results.items():
        if not trials:
            continue
        scores = [float(t.get("score", 0.0)) for t in trials]
        best_score = max(scores)
        best_idx = scores.index(best_score)
        best_params = trials[best_idx].get("params", {})

        timestamps = [t.get("_timestamp") for t in trials if t.get("_timestamp") is not None]
        if len(timestamps) >= 2:
            span_sec = (max(timestamps) - min(timestamps)) / 1000
            rate = len(trials) / span_sec if span_sec > 0 else None
        else:
            span_sec, rate = None, None

        summary.append({
            "experiment": exp,
            "trials": len(trials),
            "best_score": best_score,
            "mean": mean(scores),
            "median": median(scores),
            "std": pstdev(scores) if len(scores) > 1 else 0.0,
            "best_params": best_params,
            "duration_sec": span_sec,
            "throughput_per_sec": rate,
        })
    return summary

def save_summary_csv(rows, outfile):
    if not rows:
        return False
    fieldnames = ["experiment", "trials", "best_score", "mean", "median", "std", "duration_sec", "throughput_per_sec", "best_params"]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["best_params"] = json.dumps(row["best_params"], sort_keys=True)
            writer.writerow(out_row)
    print(f"✅ Saved summary CSV → {outfile}")
    return True

def save_trials_csv(all_results, outfile):
    rows = []
    for exp, trials in all_results.items():
        for t in trials:
            rows.append({
                "experiment": exp,
                "model": t.get("model"),
                "score": t.get("score"),
                "params": json.dumps(t.get("params", t), sort_keys=True),
                "partition": t.get("_partition"),
                "offset": t.get("_offset"),
                "timestamp_ms": t.get("_timestamp"),
                "worker_id": t.get("worker_id"),
                "duration_sec": t.get("duration_sec"),
            })
    if not rows:
        return False
    fieldnames = ["experiment", "model", "score", "params", "partition", "offset", "timestamp_ms", "worker_id", "duration_sec"]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved trial CSV → {outfile}")
    return True

def save_summary_md(rows, outfile):
    if not rows:
        return False
    with open(outfile, "w") as f:
        f.write("| Experiment | Trials | Best | Mean | Median | Std | Duration (s) | Thpt (/s) | Best Params |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            best_params = json.dumps(r["best_params"], sort_keys=True)
            duration = r["duration_sec"] if r["duration_sec"] is not None else 0.0
            throughput = r["throughput_per_sec"] if r["throughput_per_sec"] is not None else 0.0
            f.write(f"| {r['experiment']} | {r['trials']} | {r['best_score']:.3f} | {r['mean']:.3f} | {r['median']:.3f} | {r['std']:.3f} | {duration:.2f} | {throughput:.3f} | `{best_params}` |\n")
    print(f"✅ Saved summary Markdown → {outfile}")
    return True

def summarize_workers(all_results):
    stats = defaultdict(lambda: {"tasks": 0, "total_duration": 0.0, "experiments": set()})
    for exp, trials in all_results.items():
        for t in trials:
            worker = t.get("worker_id") or "unknown"
            stats[worker]["tasks"] += 1
            stats[worker]["experiments"].add(exp)
            duration = t.get("duration_sec")
            if duration is not None:
                stats[worker]["total_duration"] += float(duration)

    rows = []
    for worker, s in stats.items():
        avg = (s["total_duration"] / s["tasks"]) if s["tasks"] else 0.0
        rows.append({
            "worker_id": worker,
            "tasks": s["tasks"],
            "total_duration_sec": s["total_duration"],
            "avg_duration_sec": avg,
            "experiments": ",".join(sorted(s["experiments"])) if s["experiments"] else "",
        })
    return rows

def save_worker_csv(rows, outfile):
    if not rows:
        return False
    fieldnames = ["worker_id", "tasks", "total_duration_sec", "avg_duration_sec", "experiments"]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved worker CSV → {outfile}")
    return True

def save_worker_md(rows, outfile):
    if not rows:
        return False
    with open(outfile, "w") as f:
        f.write("| Worker | Tasks | Total Duration (s) | Avg Duration (s) | Experiments |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['worker_id']} | {r['tasks']} | {r['total_duration_sec']:.2f} | {r['avg_duration_sec']:.2f} | {r['experiments']} |\n")
    print(f"✅ Saved worker Markdown → {outfile}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Collect Kafka results_* topics and generate plots/tables.")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap servers.")
    parser.add_argument("--limit", type=int, default=50, help="Max messages to read per topic.")
    parser.add_argument("--idle-timeout", type=float, default=5.0, help="Seconds of inactivity before stopping once some data has arrived.")
    parser.add_argument("--no-wait", action="store_true", help="Exit after the first batch instead of waiting for idle timeout.")
    parser.add_argument("--from-earliest", action="store_true", help="Read from the beginning of the topic instead of only new messages.")
    parser.add_argument("--label", default=None, help="Optional run label to embed in output filenames (e.g., 'knn-3workers').")
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.label or "results"
    slug = f"{timestamp}_{label}"
    outdir = os.path.join("plots", slug)
    os.makedirs(outdir, exist_ok=True)

    results = collect_all(
        bootstrap=args.bootstrap,
        limit=args.limit,
        wait=not args.no_wait,
        idle_timeout=args.idle_timeout,
        from_latest=not args.from_earliest,
    )
    summary_rows = summarize(results)
    worker_rows = summarize_workers(results)

    # Save tables
    save_summary_csv(summary_rows, os.path.join(outdir, "summary.csv"))
    save_trials_csv(results, os.path.join(outdir, "trials.csv"))
    save_summary_md(summary_rows, os.path.join(outdir, "summary.md"))
    save_worker_csv(worker_rows, os.path.join(outdir, "workers.csv"))
    save_worker_md(worker_rows, os.path.join(outdir, "workers.md"))

    # Plots
    plot_loss_curves(results, os.path.join(outdir, "loss.png"))
    plot_accuracy_curves(results, os.path.join(outdir, "accuracy.png"))
    plot_best_bars(summary_rows, os.path.join(outdir, "best.png"))
    plot_box(results, os.path.join(outdir, "box.png"))
    plot_knn_scatter(results, os.path.join(outdir, "knn_scatter.png"))
    plot_duration_curves(results, os.path.join(outdir, "duration.png"))
    plot_worker_time(worker_rows, os.path.join(outdir, "worker_time.png"))

    print(f"🗂️ Saved run outputs under {outdir}")

if __name__ == "__main__":
    main()
