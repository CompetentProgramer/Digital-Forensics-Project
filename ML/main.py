from sklearn.datasets import make_classification # Purely for testing, remove this later
from sklearn.model_selection import train_test_split
from train import train_model, predict, save_model, load_model
from report import report_accuracy
import os

save_dir = "saved_models" # Folder where the model will be stored
save_file = "saved_model/random_forest.pkl"

def run_training():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(save_file):
        model = load_model(save_file)
    else:
        print("Training a new model...")
        model = train_model(X_train, y_train)
        save_model(model, save_dir)  # Save model

    y_pred = predict(model, X_test)
    report_accuracy(y_test, y_pred)

if __name__ == "__main__":
    run_training()

