from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    return model

def save_model(model, save_dir):
    model_path = os.path.join(save_dir, "random_forest_classifier.joblib")
    joblib.dump(model, model_path)  # For saving the Random Forest object
    print(f"Model saved to {model_path}")

def load_model(save_dir):
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Model not found at {save_dir}")

    model = joblib.load(save_dir)
    print(f"Model loaded from {save_dir}")
    return model

def predict(model, x_test):
    return model.predict(x_test)