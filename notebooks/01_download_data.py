"""step 1: grab the reddit mental health dataset from kaggle and save it into data/raw"""

import os
import shutil
import pandas as pd
import kagglehub

raw_data_dir = "data/raw"

cache_path = kagglehub.dataset_download("neelghoshal/reddit-mental-health-data")

os.makedirs(raw_data_dir, exist_ok=True)
shutil.copytree(cache_path, raw_data_dir, dirs_exist_ok=True)

df = pd.read_csv("data/raw/data_to_be_cleansed.csv")
print(df.head())
print(df.columns)
print(df.shape)
