from sklearn.metrics import accuracy_score, classification_report

def report_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}\n")
    print(f"Classification Report: {classification_report(y_true, y_pred)}\n")

