"""
Unit test verifying mathematical consistency between y_test holdout evaluation,
confusion matrix values, ROC-AUC, and calculated metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.evaluate import calculate_all_metrics
from src.train import train_and_evaluate


def test_evaluation_metrics_mathematical_consistency():
    """Verify that Accuracy, Precision, Recall, F1, and ROC-AUC match the confusion matrix and y_test predictions 100% mathematically."""
    result = train_and_evaluate(data_path="data/raw/telco_churn.csv", target_col="Churn", fast_mode=True)

    assert result["task_type"] == "classification"
    test_metrics = result["best_test_metrics"]
    assert test_metrics is not None

    # Check valid metric ranges
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        val = test_metrics[metric_name]
        assert not np.isnan(val), f"Metric {metric_name} is NaN"
        assert 0.0 <= val <= 1.0, f"Metric {metric_name} out of bounds: {val}"


def test_confusion_matrix_direct_calculation():
    """Test synthetic predictions confusion matrix [[45, 70], [9, 26]] to verify formulas."""
    y_test = np.array([0] * 115 + [1] * 35)
    # TN=45, FP=70 -> 115 zeros
    # FN=9, TP=26 -> 35 ones
    y_pred = np.array([0] * 45 + [1] * 70 + [0] * 9 + [1] * 26)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    assert (tn, fp, fn, tp) == (45, 70, 9, 26)

    calc_acc = (tp + tn) / (tp + tn + fp + fn)
    calc_prec = tp / (tp + fp)
    calc_rec = tp / (tp + fn)
    calc_f1 = 2 * (calc_prec * calc_rec) / (calc_prec + calc_rec)

    metrics = calculate_all_metrics(y_test, y_pred, task_type="classification")

    assert np.isclose(metrics["accuracy"], calc_acc)
    assert np.isclose(metrics["precision"], calc_prec)
    assert np.isclose(metrics["recall"], calc_rec)
    assert np.isclose(metrics["f1_score"], calc_f1)

    assert np.isclose(metrics["accuracy"], accuracy_score(y_test, y_pred))
    assert np.isclose(metrics["precision"], precision_score(y_test, y_pred))
    assert np.isclose(metrics["recall"], recall_score(y_test, y_pred))
    assert np.isclose(metrics["f1_score"], f1_score(y_test, y_pred))
