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

### Single sequence prediction (51 characters)

To predict methylation for a single protein sequence of exactly 51 characters:

```bash
docker build -t deeprmethylsite .
docker run --rm -v $(pwd)/results:/results deeprmethylsite src/infer.py "PKKQLILKVISGQQLPKPPDSMFGDRGEIIDPFVEVEIIGLPVDCCKDQTR"
```

The result will be displayed on screen and saved to `/results/infer_result.txt` in the container (mapped to `./results/infer_result.txt` on your host if you mount the volume).

### Long sequence prediction (multiple arginines)

To predict methylation for all arginines in a longer protein sequence:

```bash
docker build -t deeprmethylsite .
docker run --rm -v $(pwd)/results:/results deeprmethylsite src/infer.py "YOUR_LONG_PROTEIN_SEQUENCE_HERE"
```

The script will:
- Automatically detect all arginine (R) residues in the sequence
- Extract a 51-character window (25 residues before + R + 25 residues after) for each R
- Skip R residues that are too close to the sequence boundaries (< 25 residues before or after)
- Make predictions for all valid windows in batch
- Display results in a table format
- Save detailed results to `/results/infer_results.txt`

**Example:**
```bash
docker run --rm -v $(pwd)/results:/results deeprmethylsite src/infer.py "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
```

## Dataset Format

If you would like to use DeepRmethylSite to predict Arginine Methylation sites in the protein of your interest, you can provide:

1. **A single 51-character window** (25 residues before + R + 25 residues after) for precise prediction on a specific arginine
2. **A longer protein sequence** - the script will automatically process all arginines with sufficient context

The model works with window size 51, meaning for each arginine you need 25 residues upstream and 25 residues downstream. For example, if you want to predict whether the 'Arginine' residue in Position 735 in protein Q4KWH8 is methylated or not, you can provide either:
- The 51-character window: `PKKQLILKVISGQQLPKPPDSMFGDRGEIIDPFVEVEIIGLPVDCCKDQTR`
- The full protein sequence (the script will extract windows automatically)

# Citation:
Please cite the following paper if you use DeepRMethylsite.
DeepRMethylSite: Meenal Chaudhari*, Niraj Thapa*, Kaushik Roy, Robert H. Newman, Hiroto Saigo, Dukka B. KC, DeepRMethylSite: Prediction of Arginine Methylation in Proteins using Deep Learning, Mol. Omics, 2020,16, 448-454.
 # Contact 
 Feel free to contact us if you need any help : dukka.kc@wichita.edu 
 
