"""
Inference script for the DeepRMethylSite model.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALPHABET, process_single_sequence
from src.models import ensemble_final_pred, load_models


def find_arginine_positions(sequence):
    """
    Find all positions of arginine (R) in the sequence.

    Args:
        sequence: Protein sequence string

    Returns:
        List of positions (0-indexed) where R is found
    """
    positions = []
    for i, char in enumerate(sequence):
        if char == "R":
            positions.append(i)
    return positions


def extract_windows_for_arginines(sequence):
    """
    Extract 51-character windows for each arginine that has enough context.

    Args:
        sequence: Protein sequence string (any length)

    Returns:
        List of tuples: (r_position, window_sequence)
        - r_position: Position of R in original sequence (0-indexed)
        - window_sequence: 51-character window centered on R
    """
    windows = []
    r_positions = find_arginine_positions(sequence)

    for r_pos in r_positions:
        # Check if we have at least 25 residues before and after
        if r_pos >= 25 and r_pos + 25 < len(sequence):
            start = r_pos - 25
            end = r_pos + 26  # +1 to include the R
            window = sequence[start:end]

            # Validate window length
            if len(window) == 51:
                windows.append((r_pos, window))

    return windows


def predict_sequence(sequence, models=None, weights=None):
    """
    Predict methylation for a single protein sequence.

    Args:
        sequence: Protein sequence string (must be 51 characters)
        models: Optional pre-loaded models (to avoid reloading)
        weights: Optional ensemble weights

    Returns:
        Tuple of (prediction_class, probability)
        - prediction_class: 0 (not methylated) or 1 (methylated)
        - probability: Probability of methylation (0-1)
    """
    # Process sequence
    lstm_data, cnn_data = process_single_sequence(sequence)

    # Load models if not provided
    if models is None:
        models = load_models(verbose=False)

    if weights is None:
        weights = [0.16666667, 0.83333333]

    # Make prediction
    Y_pred = ensemble_final_pred(models, weights, lstm_data, cnn_data)
    probability = Y_pred[0, 1]  # Probability of methylation (class 1)
    prediction_class = 1 if probability > 0.5 else 0

    return prediction_class, probability


def predict_multiple_sequences(sequences, models=None, weights=None):
    """
    Predict methylation for multiple sequences (batch processing).

    Args:
        sequences: List of 51-character sequences
        models: Optional pre-loaded models
        weights: Optional ensemble weights

    Returns:
        List of tuples: (prediction_class, probability) for each sequence
    """
    if models is None:
        models = load_models(verbose=False)

    if weights is None:
        weights = [0.16666667, 0.83333333]

    # Process all sequences
    lstm_data_list = []
    cnn_data_list = []

    for seq in sequences:
        lstm_data, cnn_data = process_single_sequence(seq)
        lstm_data_list.append(lstm_data[0])
        cnn_data_list.append(cnn_data[0])

    # Convert to arrays
    lstm_batch = np.array(lstm_data_list)
    cnn_batch = np.array(cnn_data_list)

    # Make batch predictions
    Y_pred = ensemble_final_pred(models, weights, lstm_batch, cnn_batch)

    results = []
    for i in range(len(sequences)):
        probability = Y_pred[i, 1]
        prediction_class = 1 if probability > 0.5 else 0
        results.append((prediction_class, probability))

    return results


def save_results_to_file(results, sequence, output_file):
    """
    Save prediction results to a file.

    Args:
        results: List of tuples (r_position, window, prediction_class, probability)
        sequence: Original sequence
        output_file: Path to output file
    """
    with open(output_file, "w") as f:
        f.write("DeepRMethylSite Prediction Results\n")
        f.write("=" * 60 + "\n")
        f.write("Date: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Original sequence: {}\n".format(sequence))
        f.write("Sequence length: {} characters\n".format(len(sequence)))
        f.write("Number of arginines analyzed: {}\n".format(len(results)))
        f.write("=" * 60 + "\n\n")

        f.write(
            "{:<8} {:<10} {:<15} {:<12}\n".format(
                "R Pos", "Sequence", "Prediction", "Probability"
            )
        )
        f.write("-" * 60 + "\n")

        for r_pos, window, pred_class, prob in results:
            prediction_str = "METHYLATED" if pred_class == 1 else "NOT METHYLATED"
            f.write(
                "{:<8} {:<10} {:<15} {:<12.4f}\n".format(
                    r_pos + 1,  # 1-indexed for readability
                    window,
                    prediction_str,
                    prob,
                )
            )

        f.write("\n" + "=" * 60 + "\n")


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

    # Validate characters
    for char in sequence:
        if char not in ALPHABET:
            print(
                "Error: Invalid character '{}' in sequence. Valid characters: {}".format(
                    char, ALPHABET
                )
            )
            sys.exit(1)

    try:
        # Create results directory if it doesn't exist
        results_dir = "/results"
        os.makedirs(results_dir, exist_ok=True)

        # Check sequence length
        if len(sequence) == 51:
            # Single sequence prediction (original behavior)
            models = load_models(verbose=False)
            prediction_class, probability = predict_sequence(sequence, models=models)

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

            # Save to file
            output_file = os.path.join(results_dir, "infer_result.txt")
            with open(output_file, "w") as f:
                f.write("DeepRMethylSite Prediction Result\n")
                f.write("=" * 50 + "\n")
                f.write("Sequence: {}\n".format(sequence))
                f.write("Length: {} characters\n".format(len(sequence)))
                f.write(
                    "Prediction: {}\n".format(
                        "METHYLATED" if prediction_class == 1 else "NOT METHYLATED"
                    )
                )
                f.write("Probability of methylation: {:.4f}\n".format(probability))
                f.write("=" * 50 + "\n")
            print("Results saved to: {}".format(output_file))

        else:
            # Multiple R prediction
            windows = extract_windows_for_arginines(sequence)

            if not windows:
                print(
                    "No arginines found with sufficient context (need 25 residues before and after each R)"
                )
                print(
                    "Sequence length: {}, Number of R found: {}".format(
                        len(sequence), len(find_arginine_positions(sequence))
                    )
                )
                sys.exit(1)

            print("\n" + "=" * 60)
            print("DeepRMethylSite Prediction - Multiple Arginines")
            print("=" * 60)
            print("Original sequence: {}".format(sequence))
            print("Sequence length: {} characters".format(len(sequence)))
            print(
                "Total arginines found: {}".format(
                    len(find_arginine_positions(sequence))
                )
            )
            print("Arginines with sufficient context: {}".format(len(windows)))
            print("=" * 60 + "\n")

            # Load models once
            models = load_models(verbose=False)
            weights = [0.16666667, 0.83333333]

            # Extract windows for batch processing
            window_sequences = [win for _, win in windows]

            # Make batch predictions
            results_batch = predict_multiple_sequences(
                window_sequences, models=models, weights=weights
            )

            # Prepare results for display and saving
            results = []
            for (r_pos, window), (pred_class, prob) in zip(windows, results_batch):
                results.append((r_pos, window, pred_class, prob))

            # Display results
            print(
                "{:<8} {:<10} {:<15} {:<12}".format(
                    "R Pos", "Sequence", "Prediction", "Probability"
                )
            )
            print("-" * 60)

            for r_pos, window, pred_class, prob in results:
                prediction_str = "METHYLATED" if pred_class == 1 else "NOT METHYLATED"
                print(
                    "{:<8} {:<10} {:<15} {:<12.4f}".format(
                        r_pos + 1,  # 1-indexed for readability
                        window,
                        prediction_str,
                        prob,
                    )
                )

            print("=" * 60 + "\n")

            # Save results to file
            output_file = os.path.join(results_dir, "infer_results.txt")
            save_results_to_file(results, sequence, output_file)
            print("Results saved to: {}".format(output_file))

    except ValueError as e:
        print("Error: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("Unexpected error: {}".format(e))
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
