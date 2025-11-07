"""
Test script for the DeepRMethylSite model.
"""

import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
)

from src.data import load_test_data
from src.models import ensemble_final_pred, load_models


def main():
    """Main test function."""
    # Load test data
    r_test_x, r_test_y, r_test_x2, r_test_y2 = load_test_data()

    # Load models
    models = load_models()

    # Ensemble weights (from grid search)
    weights = [0.16666667, 0.83333333]

    # Make predictions
    Y_pred = ensemble_final_pred(models, weights, r_test_x, r_test_x2)
    t_pred2 = Y_pred[:, 1]
    Y_pred = Y_pred > 0.5
    y_pred1 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
    y_pred1 = np.array(y_pred1)

    # Calculate metrics
    print("Matthews Correlation : ", matthews_corrcoef(r_test_y, y_pred1))
    print("Confusion Matrix : \n", confusion_matrix(r_test_y, y_pred1))

    # For sensitivity and specificity
    cm = confusion_matrix(r_test_y, y_pred1)
    tn, fp = cm[0]
    fn, tp = cm[1]
    sp_2 = tn / (tn + fp)
    sn_2 = tp / (tp + fn)

    # ROC
    fpr, tpr, _ = roc_curve(r_test_y, t_pred2)
    roc_auc = auc(fpr, tpr)
    print("AUC : ", roc_auc)
    print(classification_report(r_test_y, y_pred1))

    # Create results directory if it doesn't exist
    results_dir = "/results"
    os.makedirs(results_dir, exist_ok=True)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for ST")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "image.png"))

    print("Specificity = ", sp_2, " Sensitivity = ", sn_2)


if __name__ == "__main__":
    main()
