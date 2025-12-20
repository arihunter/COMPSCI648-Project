import torch

def binary_predictions(scores):
    """
    Convert continuous scores to {-1, +1}
    """
    return torch.where(scores >= 0, 1.0, -1.0)

def accuracy(y_true, y_pred):
    """Binary accuracy for {-1,+1} labels."""
    return (y_true == y_pred).float().mean().item()

def precision(y_true, y_pred):
    """Positive predictive value for {-1,+1} labels."""
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == -1)).sum().item()
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred):
    """True positive rate for {-1,+1} labels."""
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred == -1) & (y_true == 1)).sum().item()
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)

def roc_auc_score(y_true, scores):
    """
    Manual ROC-AUC (no sklearn)
    y_true ∈ {-1, +1}
    scores ∈ ℝ
    """
    # sort by descending score
    sorted_idx = torch.argsort(scores, descending=True)
    y_true = y_true[sorted_idx]

    pos = (y_true == 1).sum().item()
    neg = (y_true == -1).sum().item()

    if pos == 0 or neg == 0:
        return 0.0

    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    prev_fpr = 0.0

    for label in y_true:
        if label == 1:
            tpr += 1 / pos
        else:
            fpr += 1 / neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return auc
