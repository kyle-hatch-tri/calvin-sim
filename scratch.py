# import os 
# import time
# from tqdm import tqdm, trange

# print('os.getenv("CUDA_VISIBLE_DEVICES"):', os.getenv("CUDA_VISIBLE_DEVICES"))
# for i in range(10):
#     time.sleep(0.2)
#     print(f'[{i}] os.getenv("CUDA_VISIBLE_DEVICES"):', os.getenv("CUDA_VISIBLE_DEVICES"))


import datetime
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
print("timestamp:", timestamp)
import ipdb; ipdb.set_trace()