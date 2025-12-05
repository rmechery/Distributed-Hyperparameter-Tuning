Distributed Hyperparameter Tuning
=================================

Kafka-backed toy framework for fanning out hyperparameter search jobs, consuming them with lightweight ML workers, and collecting the resulting metrics for comparison. Experiments ship jobs into Kafka topics, trainers pick them up and run scikit-learn models on the Iris dataset, and an evaluator plots loss curves for every experiment.

Project Layout
--------------

- `src/main.py` – produces random search configs for `knn`, `dtree`, or `logreg` experiments.
- `src/consumer.py` – Kafka worker that trains the requested model (`models.py`) and streams results back.
- `src/evaluate.py` – reads `results_*` topics, reports best accuracies, and saves `loss_curves.png`.
- `src/experiments/` – experiment definitions (`search_space()` + topic names).
- `src/search.py` – param samplers for generating random hyperparameter trials.

Prerequisites
-------------

1. Kafka + Zookeeper – install and make sure both services are running locally.
2. Python ≥ 3.9 with `pip`.
3. Project deps: `pip install -r requirements.txt`.

Quick Start
-----------

1. **Start infrastructure**
   ```bash
   # Separate terminals / background
   zookeeper-server-start.sh config/zookeeper.properties
   kafka-server-start.sh config/server.properties
   ```

2. **Launch a worker (listen + train)**
   ```bash
   python src/consumer.py
   ```

3. **Kick off an experiment producer** – run in another terminal, pick one model:
   ```bash
   python src/main.py [knn|dtree|logreg] \
       -t [optional_num_trials] \
       --dataset [iris|wine|digits|breast_cancer]
   ```
   This publishes randomly sampled configs into `hyperparams_<exp>` topics. Topics are auto-created with a few partitions to let multiple workers share the load.

4. **Evaluate + plot** – after some trials arrive, run:
   ```bash
   python src/evaluate.py
   ```
   The script scans all `results_*` topics, prints trial stats, and saves `loss_curves.png`.

How It Works
------------

1. `main.py` loads an experiment module (`experiments/knn.py`, etc.), samples configs via `random_search` (deduplicated for discrete spaces), attaches the requested dataset name, ensures the Kafka topics exist with the requested partition count, and pushes the jobs to Kafka (`hyperparams_*` topics).
2. `consumer.py` subscribes to every `hyperparams_*` topic, trains the requested model (KNN, Decision Tree, Logistic Regression) on the specified dataset, and emits results into `results_*`.
3. `evaluate.py` consumes results topics from the beginning, reports the best accuracy per experiment, and generates a loss plot for quick visual comparison.

Customization Ideas
-------------------

- Extend `src/experiments/` with new search spaces or models.
- Point `main.py`, `consumer.py`, and `evaluate.py` at remote Kafka brokers via the `bootstrap` parameter if running on multiple machines.
- Swap out the dataset in `models.py` or plug in more advanced evaluators.

AI Disclosure
-------------------
We used Github Copilot from VSCode to assist with coding, and ChatGPT for documentation and testing in this assignment.
