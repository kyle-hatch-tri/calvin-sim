import os 
import json 
import numpy as np 

# Original, normal thresholds 
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/results_oracle_goals_calvin/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.03.20_15.06.36"
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/results_oracle_goals_calvin/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/fullsusie_eval_originalthresholds_2024.04.01_00.02.31"


# Original, strict thresholds?
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.03.19_18.36.39"
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/results_oracle_goals_calvin/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/fullsusie_strictthresholds_2024.03.26_08.43.18"
RESULTS_DIR = "/home/kylehatch/Desktop/hidql/results_oracle_goals_calvin/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/fullsusie_collect_strictthresholds_2024.03.26_11.47.26"

# Original, loose thresholds?
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.03.20_15.10.17"

# # Replay w generated goals, from original thresholds to original thresholds 
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.03.25_12.32.57"

# # Replay w generated goals, from strict thresholds to original thresholds 
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.03.25_12.49.56"

# # Replay w generated goals, from loose thresholds to original thresholds 
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.03.25_13.05.28"


"/home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000/chatgptdummy16slibs22024.04.26_12.27.01"


skip_tasks = ["lift_red_block_drawer", "lift_blue_block_drawer", "lift_pink_block_drawer", "place_in_slider", "place_in_drawer", "stack_block", "unstack_block", "push_pink_block_right"]

with open(os.path.join(RESULTS_DIR, "results.json"), "r") as f:
    results = json.load(f)


    results = results["0"]

    avg_seq_len = results["avg_seq_len"]
    chain_sr = results["chain_sr"]
    task_info = results["task_info"]

    print("avg_seq_len:", avg_seq_len)

    print("\nchain_sr:")
    for key, val in chain_sr.items():
        print(f"{key}: {val}")

    new_totals = []
    print("\ntask_info")
    for task_name, info in task_info.items():
        if task_name in skip_tasks:
                continue
        
        success = info["success"]
        total = info["total"]
        success_percent = (success / total) * 100
        new_totals.append(success_percent)
        print(f"{task_name}: {success_percent:.2f}%")

    print(f"New total mean: {np.mean(new_totals):.2f}%")