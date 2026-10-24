from sklearn.metrics import accuracy_score, classification_report

def report_accuracy(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Accuracy: {accuracy}\n")
    print(f"Classification Report: {classification_report(y_test, y_pred)}\n")

