from sklearn.datasets import make_classification # Purely for testing, remove this later
from sklearn.model_selection import train_test_split
from train import train_model, predict
from report import report_accuracy
import joblib
import os

save_dir = "saved_models" # Folder where the model will be stored
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def run_training():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, X_test, y_test)

    y_pred = predict(model, X_test)

    report_accuracy(y_test, y_pred)

    model_path = os.path.join(save_dir, "random_forest_classifier.joblib")
    joblib.dump(model, model_path) # For saving the Random Forest object

if __name__ == "__main__":
    run_training()

