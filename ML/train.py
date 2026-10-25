from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model

def save_model(model, model_path):
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, model_path)  # For saving the Random Forest object
    print(f"Model saved to {model_path}")

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def predict(model, x_test):
    return model.predict(x_test)