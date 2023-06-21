# AMLS_excercise
Architecture of Machine Learning Systems 

1. Data acquisition, alignment and preparation pipline:
- `download_data` - remember to change numbers!
- `decompress data`
- `align_data` - aligns data and saves it into `data/aligned` folder
- `process_coords` - takes coords fits files and r band from `data/aligned` folder -> maps one into another -> saves result in `data/coords/*.csv` files. THIS TAKES SIGNIFICANT AMOUNT OF TIME.
  - now use `split_data` from data_preparation. This will take the data, split it based on label distribution and save it in `splitted/[test/train/validation]` dirs
- `materialize_target` - takes `data/splitted/[test/train/validation]` and `data/coords` folders -> creates two images: target and marked -> saves them into `data/splitted/[test/train/validation]/targets` folders
  - now use `split_into_smaller_pieces` from data_preparation - splits each image into pieces and stores them in `data/splitted/[test/train/validation]/[inputs/target]_pieces`

Bonus: `other/delete_files.py` to delete compressed files if they are no longer needed
 2. TODO