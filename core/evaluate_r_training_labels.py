from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_r_training_labels(true_labels, r_training_idx, r_training_pred, dataset):
    y_true = true_labels[r_training_idx]
    y_pred = r_training_pred

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== Evaluation of R-Training Labeled Nodes on {dataset.upper()} ===")
    print(f"Accuracy (new labels only): {acc:.4f}")
    print(f"Macro-F1 (new labels only): {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"R-Training Added Labels: {len(r_training_idx)}")
    return acc, f1
