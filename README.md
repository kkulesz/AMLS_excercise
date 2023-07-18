# AMLS_excercise
Architecture of Machine Learning Systems 

Project structure corresponds the instruction:
1. `data_acquisition_and_alignment`
2. `data_preparation`
3. `modeling_and_tuning`
4. `data_augmentation`

Each of these directories consists of the code used in this stage, so it is easier to see what have been done in each of them.

Rest of the directories are just by-products of implementation

### In order to train your own model or repeat the results:
1. Install libraries listed in requirements.txt
`pip install -r requirements.txt`
2. Set hyperparameters, paths, etc. in `consts.py`
3. To prepare the data needed for experiments run `scripts/data_pipeline.py`. This script constst of:
 - downloading data
 - decompressing it
 - aligning spectral bands
 - reading coordinates of galaxies and stars
 - splitting data into test/train/validation sets
 - materializing target images
 - splitting images into smaller pieces, so they fit in CUDA memory
4. Run:
 - `modeling_and_tuning/tune.py` - to choose the best set of hyperparameters
 - `modeling_and_tuning/train.py` - to train model
 - `data_augmentation/train_with_augmented_data.py` - to train model with data augmentation techniques

**Note**: you need WeightsAndBiases key to run most of the scripts provided.