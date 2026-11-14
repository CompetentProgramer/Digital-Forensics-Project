from data_loader import load_data
from train import train_model, predict, save_model, load_model
from report import report_accuracy
import os

BASE = os.path.dirname(os.path.abspath(__file__))  # paths work from any launch folder
save_dir = os.path.join(BASE, "saved_models")  # Folder where the model will be stored
save_file = os.path.join(save_dir, "random_forest_classifier.joblib")
training_file = os.path.join(BASE, "dataset", "training_augmented.csv")
validation_file = os.path.join(BASE, "dataset", "validating.csv")

def run_training():
    X_train, y_train = load_data(training_file)

    X_test, y_test = load_data(validation_file)

    if os.path.exists(save_file):
        model = load_model(save_file)
    else:
        print("Training a new model...")
        model = train_model(X_train, y_train)
        save_model(model, save_file)  # Save model

    y_pred = predict(model, X_test)
    report_accuracy(y_test, y_pred)

    save_model(model, save_file)

if __name__ == "__main__":
    run_training()