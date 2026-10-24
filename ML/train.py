from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return model

def predict(model, x_test):
    return model.predict(x_test)