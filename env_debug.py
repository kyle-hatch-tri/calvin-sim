import os 
import numpy as np 
from pathlib import Path

from calvin_env.envs.play_table_env import get_env



dataset_path = "mini_dataset"
def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    print("Before get_env")
    env = get_env(val_folder, show_gui=False)
    print("After get_env")

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


env = make_env(dataset_path)

import ipdb; ipdb.set_trace()