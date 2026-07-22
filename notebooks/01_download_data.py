"""step 1: grab the reddit mental health dataset from kaggle and save it into data/raw"""

# os lets us make folders
import os
# shutil lets us copy files and folders
import shutil
# pandas so we can peek at the csv after downloading it
import pandas as pd
# kagglehub is the library that actually talks to kaggle and downloads stuff
import kagglehub

# where we want the data to end up
raw_data_dir = "data/raw"

# download the dataset, kagglehub hands back where it saved everything
cache_path = kagglehub.dataset_download("neelghoshal/reddit-mental-health-data")

# make sure our folder exists
os.makedirs(raw_data_dir, exist_ok=True)
# copy everything from the kaggle cache into our project folder
shutil.copytree(cache_path, raw_data_dir, dirs_exist_ok=True)

# load the csv so we can see what we got
df = pd.read_csv("data/raw/data_to_be_cleansed.csv")
# show the first few rows
print(df.head())
# show what columns we have
print(df.columns)
# show how big the dataset is
print(df.shape)
