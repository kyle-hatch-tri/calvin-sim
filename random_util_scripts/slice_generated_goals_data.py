import os 
import numpy as np 
from glob import glob 
from tqdm import tqdm, trange

INPUT_DATA_DIR = ""

OUTPUT_DATA_DIR = ""


N_SAMPLES = 4 

tf_record_files = glob(os.path.join(INPUT_DATA_DIR, "**", "*.tfrecord"), recursive=True)

for tf_record_file in tqdm(tf_record_files):
    