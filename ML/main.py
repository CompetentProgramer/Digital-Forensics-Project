from sklearn.datasets import make_classification # Purely for testing, remove this later
from sklearn.model_selection import train_test_split
from train import train_model
from report import report_accuracy


def run_training():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = train_model(X_train, y_train, X_test, y_test)

    report_accuracy(y_pred, y_test)

if __name__ == "__main__":
    run_training()

