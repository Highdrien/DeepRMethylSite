"""
Inference script for the DeepRMethylSite model.
"""

import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import process_single_sequence
from src.models import ensemble_final_pred, load_models


def predict_sequence(sequence):
    """
    Predict methylation for a single protein sequence.

    Args:
        sequence: Protein sequence string (must be 51 characters)

    Returns:
        Tuple of (prediction_class, probability)
        - prediction_class: 0 (not methylated) or 1 (methylated)
        - probability: Probability of methylation (0-1)
    """
    # Process sequence
    lstm_data, cnn_data = process_single_sequence(sequence)

    # Load models (without verbose output)
    models = load_models(verbose=False)

    # Ensemble weights (from grid search)
    weights = [0.16666667, 0.83333333]

    # Make prediction
    Y_pred = ensemble_final_pred(models, weights, lstm_data, cnn_data)
    probability = Y_pred[0, 1]  # Probability of methylation (class 1)
    prediction_class = 1 if probability > 0.5 else 0

    return prediction_class, probability


def main():
    """Main inference function."""
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <protein_sequence>")
        print(
            "Example: python src/infer.py PKKQLILKVISGQQLPKPPDSMFGDRGEIIDPFVEVEIIGLPVDCCKDQTR"
        )
        sys.exit(1)

    sequence = sys.argv[1].upper().strip()

    # Remove whitespace if any
    sequence = sequence.replace(" ", "").replace("\n", "").replace("\t", "")

    try:
        prediction_class, probability = predict_sequence(sequence)

        print("\n" + "=" * 50)
        print("DeepRMethylSite Prediction Result")
        print("=" * 50)
        print("Sequence: {}".format(sequence))
        print("Length: {} characters".format(len(sequence)))
        print(
            "\nPrediction: {}".format(
                "METHYLATED" if prediction_class == 1 else "NOT METHYLATED"
            )
        )
        print("Probability of methylation: {:.4f}".format(probability))
        print("=" * 50 + "\n")

    except ValueError as e:
        print("Error: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("Unexpected error: {}".format(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
