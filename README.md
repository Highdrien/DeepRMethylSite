# DeepRMethylSite: Prediction of Arginine Methylation in Proteins using Deep Learning

DeepRMethylSite is a deep-learning based method for Arginine Methylation sites prediction in proteins. It is implemented using Keras (version 2.2.4) and Tensorflow (version 1.15) backend and has been tested on both in Windows and Linux OS. 

# Pre-requisites
  Python 3.6<br/>
  Tensorflow (>version 1.15) - (Optional: Tensorflow GPU for GPU based machine)<br/>
  Keras (>version 2.2.4) - (Optional: Keras GPU for GPU based machine)<br/>
  Numpy <br/>
  Biopython <br/>
  Sklearn <br/>
  Imblearn <br/>
  Winrar or any compression program to open .rar file
  
 # Running on CPU or GPU
 To run in CPU, installation of Tensorflow and Keras will suffice. However, to run in GPU, further Tensorflow-gpu and keras-gpu must be installed. Tensorflow GPU and Keras GPU version utilizes cuda cores in our GPU (in our case NVIDIA 2080 TI) for faster training time. However, running in GPU is not mandatory.
 
 # Dataset
 Dataset is in fasta format. Both training and testing datasets are provided which are independent (one does not include others).
 Training dataset for positive and negative are data/train/train_s33_Pos_51.fasta and data/train/train_s33_Neg_51.fasta respectively. Testing dataset for positive and negative are data/test/test_s33_Pos_51.fasta and data/test/test_s33_Neg_51.fasta respectively. Training dataset is made available so that future models can be trained for the comparison purpose.
 # Model
 The best trained model for both CNN and LSTM (used in our final results) have also been included. The models/weights/model_best_cnn.h5 is the best trained model for CNN and models/weights/model_best_lstm.h5 is the best trained model for LSTM respectively. 
 # Code
The code has been refactored into a modular structure:
     - src/data.py: Data loading and preprocessing functions
     - src/models.py: Model loading and ensemble prediction functions
     - src/test.py: Main test script (replaces model_gridsearch_load.py)
The test script requires following files to run:
     - models/weights/model_best_cnn.h5
     - models/weights/model_best_lstm.h5
     - data/test/test_s33_Pos_51.fasta
     - data/test/test_s33_Neg_51.fasta
The output of this is the result mentioned in our research paper.
# Prediction for given  test dataset (Procedure)
     - Download all rar files from models/compressed_models/models.part01.rar to models/compressed_models/models.part07.rar and keep in the same folder.
     - Open models/compressed_models/models.part01.rar with compression tool like winrar and extract it. This will extract both model files from our 
       research, model_best_cnn.h5 and model_best_lstm.h5 to models/weights/.
     - The test datasets are already in data/test/ directory.
     - Run the test script: $python3 src/test.py
     - Alternatively, use Docker $docker build -t deeprmethylsite . && docker run --rm -v $(pwd)/results:/results deeprmethylsite
# Prediction for your dataset

## Using Docker (Recommended)

To predict methylation for a single protein sequence (51 characters):

**With docker directly:**
```bash
docker build -t deeprmethylsite .
docker run --rm deeprmethylsite src/infer.py "PKKQLILKVISGQQLPKPPDSMFGDRGEIIDPFVEVEIIGLPVDCCKDQTR"
```

## Dataset Format

 If you would like to use DeepRmethylSite to predict Arginine Methylation sites in the protein of your interest, you should prepare your dataset in the same format as the test dataset which is basically a FASTA format. This model works for window size 51 only, meaning for the residue of your interest you should provide 25 resiudes downstream and 25 residues upstream. e.g. if you want to predict whether the 'Arginine' residue in Position 735 in protein Q4KWH8 is methylated or not, the input file should contain 25 residues upstream of R (position 735 in protein Q4KWH8) and 25 residues downstream of R.
 
 The general format for your dataset should be:

>sp|Q4KWH8|PLCH1_HUMAN%730%755<br/>
PKKQLILKVISGQQLPKPPDSMFGDRGEIIDPFVEVEIIGLPVDCCKDQTR

# Citation:
Please cite the following paper if you use DeepRMethylsite.
DeepRMethylSite: Meenal Chaudhari*, Niraj Thapa*, Kaushik Roy, Robert H. Newman, Hiroto Saigo, Dukka B. KC, DeepRMethylSite: Prediction of Arginine Methylation in Proteins using Deep Learning, Mol. Omics, 2020,16, 448-454.
 # Contact 
 Feel free to contact us if you need any help : dukka.kc@wichita.edu 
 
