import os 
from glob import glob 
import boto3
import time
import subprocess
import awswrangler as wr
from smart_open import smart_open
import json
import numpy as np
from calvin_agent.evaluation.utils import count_success
from collections import Counter, defaultdict
import pickle

"""
cd /home/kylehatch/Desktop/hidql/eval_outputs
aws s3 sync --exclude "*" --include "*results.txt" s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model .

aws s3 ls s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/**.txt 
"""


# S3_DIRS = ["s3://kyle-sagemaker-training-outputs/eval-outputs/el_trasho/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.02.29_09.46.15",
#            "s3://kyle-sagemaker-training-outputs/eval-outputs/el_trasho/calvin/test1_400smthlibs1_2024.02.22_19.59.48/40000/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000/2024.02.29_09.46.17"]

def view_results(args):
    seeds_dict = defaultdict(list)


    for s3_dir in args.s3_dirs:
        # checkpoint = int(s3_dir.split("/")[-2].split("_")[-1])
        # seed = int(s3_dir.split("/")[-4].split("_")[-1])
        checkpoint = s3_dir.split("/")[-2]
        seed = s3_dir.split("/")[-4]
        policy = s3_dir.split(seed)[0]
        seeds_dict[(policy, checkpoint)].append((seed, s3_dir))

    if "calvin" in args.s3_dirs[0]:
        print_results_calvin(seeds_dict)
    elif "libero" in args.s3_dirs[0]:
        print_results_libero(seeds_dict)
    else:
        raise ValueError(f"Unsupported environment: \"{args.s3_dirs}\".")
    
# def print_results_libero(seeds_dict):
#     for (policy, checkpoint), seeds_s3_dirs in seeds_dict.items():
#         avg_results = defaultdict(list)

#         avg_overall_success = []
#         seeds, s3_dirs = zip(*seeds_s3_dirs)
#         for seed, s3_dir in zip(seeds, s3_dirs):
#             results_file = os.path.join(os.path.join(s3_dir, "results.json"))
#             with smart_open(results_file, 'rb') as s3_source:
#                 results = json.load(s3_source)
#                 for task_name, successes in results.items():
#                     # print(f"{task_name}: {np.mean(successes) * 100:.1f}%")
#                     avg_results[task_name].append(np.mean(successes))
#                     avg_overall_success.append(np.mean(successes))
#                 avg_results["overall"].append(np.mean(avg_overall_success))
        
#         print(f"\n\nAvg results")
#         print(f"policy: {policy}")
#         print(f"checkpoint: {checkpoint}")
#         print(f"seeds: {seeds}")
#         for task_name, avg_successes in avg_results.items():
#             print(f"{task_name}: {np.mean(avg_successes) * 100:.1f}%")
    


def print_results_libero(seeds_dict):
    for (policy, checkpoint), seeds_s3_dirs in seeds_dict.items():
        avg_results = defaultdict(lambda: defaultdict(list))

        # average across seeds 
        seeds, s3_dirs = zip(*seeds_s3_dirs)
        for seed, s3_dir in zip(seeds, s3_dirs):
            results_file = os.path.join(os.path.join(s3_dir, "results.json"))
            with smart_open(results_file, 'rb') as s3_source:
                results = json.load(s3_source)
                for task_name, result in results.items():
                    for metric, value_list in result.items():
                        avg_results[task_name][metric].append(np.mean(value_list))

                        if metric == "success_sequence_length":
                            at_least_one_subtask_success = np.array(value_list) >= 1
                            at_least_two_subtask_success = np.array(value_list) >= 2
                            avg_results[task_name]["at_least_one_subtask_success"].append(np.mean(at_least_one_subtask_success))
                            avg_results[task_name]["at_least_two_subtask_success"].append(np.mean(at_least_two_subtask_success))
        
        print(f"\n\nAvg results")
        print(f"policy: {policy}")
        print(f"checkpoint: {checkpoint}")
        print(f"seeds: {seeds}")

        overall_result = defaultdict(list)
        for task_name, result in avg_results.items():
            successes = result["success"]
            at_least_one_subtask_successes = result["at_least_one_subtask_success"]
            at_least_two_subtask_successes = result["at_least_two_subtask_success"]
            success_sequence_lengths = result["success_sequence_length"]
            num_subtask_successeses = result["num_subtask_successes"] 
            # import ipdb; ipdb.set_trace()
            print(f"{task_name}: success {np.mean(successes) * 100:.1f}%, 1+ subtask success {np.mean(at_least_one_subtask_successes) * 100:.1f}%, 2+ subtask success {np.mean(at_least_two_subtask_successes) * 100:.1f}%, success_sequence_length: {np.mean(success_sequence_lengths):.3f}, num_subtask_successes: {np.mean(num_subtask_successeses):.3f}")

            overall_result["success"].append(np.mean(successes))
            overall_result["at_least_one_subtask_success"].append(np.mean(at_least_one_subtask_successes))
            overall_result["at_least_two_subtask_success"].append(np.mean(at_least_two_subtask_successes))
            overall_result["success_sequence_length"].append(np.mean(success_sequence_lengths))
            overall_result["num_subtask_successes"].append(np.mean(num_subtask_successeses))

        successes = overall_result["success"]
        at_least_one_subtask_successes = overall_result["at_least_one_subtask_success"]
        at_least_two_subtask_successes = overall_result["at_least_two_subtask_success"]
        success_sequence_lengths = overall_result["success_sequence_length"]
        num_subtask_successeses = overall_result["num_subtask_successes"]

        

        print(f"overall: success {np.mean(successes) * 100:.1f}%, 1+ subtask success {np.mean(at_least_one_subtask_successes) * 100:.1f}%, 2+ subtask success {np.mean(at_least_two_subtask_successes) * 100:.1f}%, success_sequence_length: {np.mean(success_sequence_lengths):.3f}, num_subtask_successes: {np.mean(num_subtask_successeses):.3f}")



def print_results_calvin(seeds_dict):
    for (policy, checkpoint), seeds_s3_dirs in seeds_dict.items():
        seq_lens = []
        chain_srs = defaultdict(list)

        seeds, s3_dirs = zip(*seeds_s3_dirs)
        for seed, s3_dir in zip(seeds, s3_dirs):
            
            # sequences = load_pickled_object(os.path.join(s3_dir, "sequences.pkl"))

            
                
            

            results_file = os.path.join(os.path.join(s3_dir, "results.npy"))
            with smart_open(results_file, 'rb') as s3_source:
                results = np.load(s3_source)

                seq_len = np.mean(results)
                seq_lens.append(seq_len)
                chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
                # print(f"\n\npolicy: {policy}")
                # print(f"checkpoint: {checkpoint}")
                # print(f"seeds: {seed}")

                # print(f"Average successful sequence length: {seq_len}")
                # print("Success rates for i instructions in a row:")
                for i, sr in chain_sr.items():
                    # print(f"{i}: {sr * 100:.1f}%")
                    chain_srs[i].append(sr)

        
        print(f"\n\nAvg results")
        print(f"policy: {policy}")
        print(f"checkpoint: {checkpoint}")
        print(f"seeds: {seeds}")

        print(f"Average successful sequence length: {np.mean(seq_lens)}")
        print("Success rates for i instructions in a row:")
        for i, srs in chain_srs.items():
            print(f"{i}: {np.mean(srs) * 100:.1f}%")


        
def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def load_pickled_object(file_path):
    """
    Load an object from a file using pickle.

    Parameters:
    - file_path (str): Path to the file containing the pickled object.

    Returns:
    - obj: Loaded object.
    """
    with open(file_path, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--s3_dirs", type=str, nargs="+", help="Path to the dataset root directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    view_results(args)

"""

python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/el_trasho/liberosplit3val90/test1_400smthlib_2024.02.21_06.44.06/40000/liberosplit3/gcdiffusion/auggoaldiff/seed_0/20240306_064653/checkpoint_150000/2024.03.07_17.52.12/




# s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/auggoaldiff/

aws s3 ls --recursive s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit3val90/test1_400smthlib_2024.02.21_06.44.06/40000/liberosplit3/gcdiffusion/auggoaldiff/ | grep 'results.json'

aws s3 cp --recursive s3://susie-data/libero_data_processed/libero_10/ s3://susie-data/libero_data_processed_split3/val/libero_10/
aws s3 cp --recursive s3://susie-data/libero_data_processed/libero_90/ s3://susie-data/libero_data_processed_split3/train/libero_90/
aws s3 cp --recursive s3://susie-data/libero_data_processed/libero_goal/ s3://susie-data/libero_data_processed_split3/train/libero_goal/
aws s3 cp --recursive s3://susie-data/libero_data_processed/libero_object/ s3://susie-data/libero_data_processed_split3/train/libero_object/
aws s3 cp --recursive s3://susie-data/libero_data_processed/libero_spatial/ s3://susie-data/libero_data_processed_split3/train/libero_spatial/




python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/public_model/checkpoint_only/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_3/20240321_232950/checkpoint_150000/2024.03.25_03.25.34 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/public_model/checkpoint_only/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_1/20240321_232950/checkpoint_150000/2024.03.25_03.25.34 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/public_model/checkpoint_only/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_0/20240321_232950/checkpoint_150000/2024.03.25_03.25.34 

python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/test1_400smthlib_2024.02.21_06.44.06/40000/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_3/20240321_232950/checkpoint_150000/2024.03.26_20.59.06 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/test1_400smthlib_2024.02.21_06.44.06/40000/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_0/20240321_232950/checkpoint_150000/2024.03.26_20.59.07 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/test1_400smthlib_2024.02.21_06.44.06/40000/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_0/20240321_232950/checkpoint_500000/2024.03.26_20.59.07 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/test1_400smthlib_2024.02.21_06.44.06/40000/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_1/20240321_232950/checkpoint_500000/2024.03.26_20.59.07 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvin/test1_400smthlib_2024.02.21_06.44.06/40000/calvingcbclcbc/gcdiffusion/calvin16lconlygenfrac0.75/seed_1/20240321_232950/checkpoint_150000/2024.03.26_20.59.07


python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_0/20240410_010011/checkpoint_150000/2024.04.12_23.43.53 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_1/20240410_010011/checkpoint_150000/2024.04.12_23.43.56 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_2/20240410_010011/checkpoint_150000/2024.04.12_23.44.00 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_3/20240410_010011/checkpoint_150000/2024.04.12_23.43.55


python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_0/20240410_010011/checkpoint_1000000/2024.04.12_23.44.00 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_1/20240410_010011/checkpoint_1000000/2024.04.12_23.43.55 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_2/20240410_010011/checkpoint_1000000/2024.04.12_23.43.56 \
s3://kyle-sagemaker-training-outputs/eval-outputs/calvinlcbc/public_model/checkpoint_only/calvinlcbc/lcdiffusion/auggoaldiff/seed_3/20240410_010011/checkpoint_1000000/2024.04.12_23.43.55


python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_0/20240410_004925/checkpoint_150000/2024.04.12_22.49.03 \
s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_1/20240410_004925/checkpoint_150000/2024.04.12_22.49.03 \
s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_2/20240410_004925/checkpoint_150000/2024.04.12_22.49.04 \
s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_3/20240410_004925/checkpoint_150000/2024.04.12_22.49.08


python3 -u view_eval_outputs3.py \
--s3_dirs s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_0/20240410_004925/checkpoint_1000000/2024.04.12_22.49.01 \
s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_1/20240410_004925/checkpoint_1000000/2024.04.12_22.49.08 \
s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_2/20240410_004925/checkpoint_1000000/2024.04.12_22.49.03 \
s3://kyle-sagemaker-training-outputs/eval-outputs/liberosplit2/public_model/checkpoint_only/liberosplit2/lcdiffusion/auggoaldiff/seed_3/20240410_004925/checkpoint_1000000/2024.04.12_22.49.03
"""