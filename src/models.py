from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def get_dataset():
    data = load_iris()
    return data.data, data.target

def train_model(model_name: str, params: dict):
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "logreg":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000)
        )
    elif model_name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5))
        )
    elif model_name == "dtree":
        model = DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", None))
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
