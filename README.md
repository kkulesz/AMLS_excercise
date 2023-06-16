# AMLS_excercise
Architecture of Machine Learning Systems 


1. Data acquisition and alignment pipline:
- `download_data.py` - remember to change numbers!
- `decompress data.py`
- `align_data.py` - aligns data and saves it into `data/aligned` folder
- `process_coords.py` - takes coords fits files and r band from `data/aligned` folder -> maps one into another -> saves result in `data/coords/*.csv` files
- `materialize_target.py` - takes `data/aligned` and `data/coords` folders -> creates two images: target and marked -> saves them into `data/targets` folder


Bonus: `other/delete_files.py` to delete compressed files if they are no longer needed

2. Data preparation
- `split_into_smaller_pieces.py` -  splits each image into pieces and stores them in `data/ready/[targets/inputs]`
- `split_data.py` - TODO