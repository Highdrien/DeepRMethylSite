"""
Models for the DeepRMethylSite model.
"""

import os

from keras.models import load_model
from numpy import array, tensordot


def load_models(models_dir="models/weights"):
    """
    Load LSTM and CNN models.

    Args:
        models_dir: Directory containing model weight files

    Returns:
        List of loaded models [LSTM, CNN]
    """
    lstm_path = os.path.join(models_dir, "model_best_lstm.h5")
    cnn_path = os.path.join(models_dir, "model_best_cnn.h5")

    models = [load_model(lstm_path), load_model(cnn_path)]

    for i in models:
        print(i.summary())

    return models


def ensemble_final_pred(members, weights, testX, testX2):
    """
    Make ensemble predictions using weighted combination of models.

    Args:
        members: List of model objects [LSTM, CNN]
        weights: List of weights for each model
        testX: Test data for LSTM model
        testX2: Test data for CNN model

    Returns:
        Array of weighted predictions
    """
    # make predictions
    yhats = []
    yhats.append(array(members[0].predict(testX)))
    yhats.append(array(members[1].predict(testX2)))
    yhats = array(yhats)
    # weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    # result = argmax(summed, axis=1)
    return summed
