import os 
import json 
import numpy as np 
from collections import defaultdict


# # Auggoal goal diff
# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000/2024.03.28_11.25.11"

# Default 
RESULTS_DIR = "/home/kylehatch/Desktop/hidql/results_oracle_goals_libero/single_task/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/default/seed_0/20240301_015613/checkpoint_500000/2024.03.28_11.24.24"

skip_tasks = []

with open(os.path.join(RESULTS_DIR, "results.json"), "r") as f:
    results = json.load(f)


    overall_result = defaultdict(list)
    for task_name, result in results.items():
        if task_name in skip_tasks:
            continue 

        print(f"\n{task_name}:")
        for metric, value_list in result.items():
            print(f"\t{metric}: {np.mean(value_list):.3f}")

            overall_result[metric].append(np.mean(value_list))

            if metric == "success_sequence_length":
                at_least_one_subtask_success = np.array(value_list) >= 1
                at_least_two_subtask_success = np.array(value_list) >= 2
                print(f"\tat_least_one_subtask_success: {np.mean(at_least_one_subtask_success):.3f}")
                print(f"\tat_least_two_subtask_success: {np.mean(at_least_two_subtask_success):.3f}")
                overall_result["at_least_one_subtask_success"].append(np.mean(at_least_one_subtask_success))
                overall_result["at_least_two_subtask_success"].append(np.mean(at_least_two_subtask_success))

    print("\noverall:")
    for metric, value_list in overall_result.items():
        print(f"\t{metric}: {np.mean(value_list):.3f}")


print("RESULTS_DIR:", RESULTS_DIR)