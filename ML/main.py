from data_loader import load_data
from train import train_model, predict, save_model, load_model
from report import report_accuracy
import os

save_dir = "saved_models" # Folder where the model will be stored
save_file = "saved_models/random_forest_classifier.joblib"
training_file = "dataset/training.csv"
validation_file = "dataset/validating.csv"

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

if __name__ == "__main__":
    run_training()

