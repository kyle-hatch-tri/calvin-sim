import os 
import numpy as np 
import h5py

filepath = "/home/kylehatch/Desktop/hidql/data/droid_data/susie_kyle/2024-04-24/Wed_Apr_24_16:16:06_2024/trajectory.h5"



# def load_h5_file(file_path):
#     """
#     Load an H5 file and print its content.

#     Parameters:
#     file_path (str): The path to the H5 file.

#     Returns:
#     dict: A dictionary with the contents of the H5 file.
#     """
#     try:
#         with h5py.File(file_path, 'r') as file:
#             data = {}
            
#             def visit_fn(name, node):
#                 if isinstance(node, h5py.Dataset):
#                     data[name] = node[...]
#                 elif isinstance(node, h5py.Group):
#                     data[name] = {}
            
#             file.visititems(visit_fn)
            
#             return data
#     except Exception as e:
#         print(f"An error occurred while loading the file: {e}")
#         return None
def load_h5_file(file_path):
    """
    Load an H5 file and return its content as a nested dictionary.

    Parameters:
    file_path (str): The path to the H5 file.

    Returns:
    dict: A dictionary with the nested structure of the H5 file.
    """
    def recursively_load(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[...]
            elif isinstance(item, h5py.Group):
                data[key] = recursively_load(item)
        return data

    try:
        with h5py.File(file_path, 'r') as file:
            return recursively_load(file)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

# Example usage
data = load_h5_file(filepath)
import ipdb; ipdb.set_trace()
# if data is not None:
#     for key, value in data.items():
#         print(f"{key}: {value}")
