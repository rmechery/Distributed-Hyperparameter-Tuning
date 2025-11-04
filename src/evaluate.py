import matplotlib.pyplot as plt

def plot_results(results, metric_name="Accuracy"):
    params = [str(r[0]) for r in results]
    scores = [r[1] for r in results]
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(scores)), scores, color="skyblue")
    plt.yticks(range(len(scores)), params)
    plt.xlabel(metric_name)
    plt.ylabel("Parameter Configurations")
    plt.title("Hyperparameter Search Results")
    plt.tight_layout()
    plt.show()
