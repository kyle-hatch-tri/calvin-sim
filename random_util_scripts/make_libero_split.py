import os 
from glob import glob 
import shutil 
from tqdm import tqdm, trange 


INPUT_DIR = "/home/kylehatch/Desktop/hidql/data/libero_data_processed_split2"
OUTPUT_DIR = "/home/kylehatch/Desktop/hidql/data/libero_data_processed_atmsplithighlevel"

# ATM split, where seperate policies are trained on each LIBERO suite 
# ATM split, but one policy for everything 
# libs2 split, but add 10 trajectories for the val tasks. Maybe upweight these ones or something? Or at least duplicate them so they have the same frequency as the other tasks. 

def make_data_split():
    import ipdb; ipdb.set_trace() 
    all_files = glob(os.path.join(INPUT_DIR, "**", "*.tfrecord"), recursive=True)


    train_files = [file for file in all_files]
    val_files

if __name__ == "__main__":
    make_data_split()