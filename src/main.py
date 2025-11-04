from search import random_search
from models import train_model
from evaluate import plot_results

if __name__ == "__main__":
    # Define simple search space
    search_space = {
        "n_neighbors": ("int_range", 1, 15, 2),
    }

    # Define evaluation function
    def evaluate(params):
        return train_model("knn", params)

    # Run random search
    results = random_search(search_space, evaluate, n_trials=10)

    # Display top results
    for params, score in results:
        print(f"Params: {params} -> Accuracy: {score:.3f}")

    # Plot
    plot_results(results)
