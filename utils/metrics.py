from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred, average='binary'):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0, average=average)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0, average=average)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0, average=average)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
