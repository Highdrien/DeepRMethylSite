"""
Data processing script for the DeepRMethylSite model.
"""

import os

import numpy as np
from Bio import SeqIO
from imblearn.under_sampling import RandomUnderSampler
from numpy import array
from sklearn.utils import shuffle

# Constants
ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
POSIT_1 = 1
NEGAT_0 = 0
WIN_SIZE = 51  # actual window size
WIN_SIZE1 = 39
WIN_SIZE2 = 21
NUM_CLASSES = 2
N_MODELS = 2
CUT_OFF1 = int((51 - WIN_SIZE1) / 2)
CUT_OFF2 = int((51 - WIN_SIZE2) / 2)

# define a mapping of chars to integers
CHAR_TO_INT = dict((c, i) for i, c in enumerate(ALPHABET))
INT_TO_CHAR = dict((i, c) for i, c in enumerate(ALPHABET))


def load_test_data_cnn(pos_file, neg_file):
    """
    Load and process test data for CNN model.

    Args:
        pos_file: Path to positive sequences FASTA file
        neg_file: Path to negative sequences FASTA file

    Returns:
        Tuple of (X, y) arrays for CNN model
    """
    r_test_x2 = []
    r_test_y2 = []

    # for positive sequence
    def process_positive(seq_record):
        data = seq_record.seq
        data = data[CUT_OFF1:-CUT_OFF1]
        # integer encode input data
        for char in data:
            if char not in ALPHABET:
                return
        integer_encoded = [CHAR_TO_INT[char] for char in data]
        r_test_x2.append(integer_encoded)
        r_test_y2.append(POSIT_1)

    for seq_record in SeqIO.parse(pos_file, "fasta"):
        process_positive(seq_record)

    # for negative sequence
    def process_negative(seq_record):
        data = seq_record.seq
        data = data[CUT_OFF1:-CUT_OFF1]
        # integer encode input data
        for char in data:
            if char not in ALPHABET:
                return
        integer_encoded = [CHAR_TO_INT[char] for char in data]
        r_test_x2.append(integer_encoded)
        r_test_y2.append(NEGAT_0)

    for seq_record in SeqIO.parse(neg_file, "fasta"):
        process_negative(seq_record)

    # Changing to array (matrix)
    r_test_x2 = array(r_test_x2)
    r_test_y2 = array(r_test_y2)

    # Balancing test dataset by undersampling
    rus = RandomUnderSampler(random_state=7)
    x_res4, y_res4 = rus.fit_resample(r_test_x2, r_test_y2)
    # Shuffling
    r_test_x2, r_test_y2 = shuffle(x_res4, y_res4, random_state=7)
    r_test_x2 = np.array(r_test_x2)
    r_test_y2 = np.array(r_test_y2)

    return r_test_x2, r_test_y2


def load_test_data_lstm(pos_file, neg_file):
    """
    Load and process test data for LSTM model.

    Args:
        pos_file: Path to positive sequences FASTA file
        neg_file: Path to negative sequences FASTA file

    Returns:
        Tuple of (X, y) arrays for LSTM model
    """
    r_test_x = []
    r_test_y = []

    # for positive sequence
    def process_positive(seq_record):
        data = seq_record.seq
        data = data[CUT_OFF2:-CUT_OFF2]
        # integer encode input data
        for char in data:
            if char not in ALPHABET:
                return
        integer_encoded = [CHAR_TO_INT[char] for char in data]
        r_test_x.append(integer_encoded)
        r_test_y.append(POSIT_1)

    for seq_record in SeqIO.parse(pos_file, "fasta"):
        process_positive(seq_record)

    # for negative sequence
    def process_negative(seq_record):
        data = seq_record.seq
        data = data[CUT_OFF2:-CUT_OFF2]
        # integer encode input data
        for char in data:
            if char not in ALPHABET:
                return
        integer_encoded = [CHAR_TO_INT[char] for char in data]
        r_test_x.append(integer_encoded)
        r_test_y.append(NEGAT_0)

    for seq_record in SeqIO.parse(neg_file, "fasta"):
        process_negative(seq_record)

    # Changing to array (matrix)
    r_test_x = array(r_test_x)
    r_test_y = array(r_test_y)

    # Balancing test dataset by undersampling
    rus = RandomUnderSampler(random_state=7)
    x_res3, y_res3 = rus.fit_resample(r_test_x, r_test_y)
    # Shuffling
    r_test_x, r_test_y = shuffle(x_res3, y_res3, random_state=7)
    r_test_x = np.array(r_test_x)
    r_test_y = np.array(r_test_y)

    return r_test_x, r_test_y


def load_test_data(data_dir="data/test"):
    """
    Load all test data for both CNN and LSTM models.

    Args:
        data_dir: Directory containing test FASTA files

    Returns:
        Tuple of (lstm_x, lstm_y, cnn_x, cnn_y)
    """
    pos_file = os.path.join(data_dir, "test_s33_Pos_51.fasta")
    neg_file = os.path.join(data_dir, "test_s33_Neg_51.fasta")

    lstm_x, lstm_y = load_test_data_lstm(pos_file, neg_file)
    cnn_x, cnn_y = load_test_data_cnn(pos_file, neg_file)

    return lstm_x, lstm_y, cnn_x, cnn_y
