from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def report_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}\n")

    classification = classification_report(y_true, y_pred)
    print(f"Classification Report: {classification}\n")

    confusion = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix: \n{confusion}\n")

