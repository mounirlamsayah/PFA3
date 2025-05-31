# src/config.py

import os

# Base paths
#BASE_DIR = "./data"
DATA_DIR = "./data"

# Raw data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
IMG_DIR = os.path.join(RAW_DATA_DIR, 'small_train_data_set')
CSV_PATH = os.path.join(RAW_DATA_DIR, 'train_data.csv')

# Image parameters
IMG_SIZE = (299, 299, 3)
BATCH_SIZE = 128
