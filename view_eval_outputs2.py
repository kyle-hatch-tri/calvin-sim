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
from collections import Counter


"""
cd /home/kylehatch/Desktop/hidql/eval_outputs
aws s3 sync --exclude "*" --include "*results.txt" s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model .

aws s3 ls s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/**.txt 
"""


# # ALGOS = ["public_checkpoint", "gcbc_diffusion", "gcbc", "gciql", "gciql2", "gciql3", "gciql4", "gciql5"]
# # CHECKPOINTS = ["only_checkpoint", 50_000]
# # NUM_SAMPLES = [1]
# # ALGO = "gcbc_diffusion_b2048"
# # ALGOS = ["gcbc_diffusion", "gcbc_diffusion_b2048", "gcbc_diffusion_b4096"]
# # ALGOS = ["gcbc_diffusion_20240121_214528", "gcbc_diffusion_b2048_20240131_014354", "gcbc_diffusion_b4096_20240131_014541"]
# ALGOS = ["gcbc_diffusion_20240121_214528", "gcbc_diffusion_b2048_20240131_014354", "gcbc_diffusion_b4096_20240131_014541", "gcbc_diffusion_nam_20240201_014718", "gcbc_diffusion_nam_b2048_20240201_014619", "gcbc_diffusion_nam_b4096_20240201_014758"]
# VFS = ["no_vf"]
# VF_CHECKPOINTS = ["no_checkpoint"]
# CHECKPOINTS = [50_000, 60_000]
# NUM_SAMPLES = [1]
# ALGOS = ["public_checkpoint"]
# VFS = ["no_vf"]
# VF_CHECKPOINTS = ["no_checkpoint"]
# CHECKPOINTS = ["only_checkpoint"]
# NUM_SAMPLES = [1]
ALGOS = ["gcbc_diffusion_nam_20240201_014718"]
VFS = ["no_vf"]
VF_CHECKPOINTS = ["none"]
CHECKPOINTS = [1020000, 140000, 1600000, 1940000, 20000, 40000, 490000, 60000, 600000]
NUM_SAMPLES = [1]


def view_results():
    results_files = []
    results_json_files = []
    for algo in ALGOS:
        for checkpoint in CHECKPOINTS:
            for vf in VFS:
                for vf_checkpoint in VF_CHECKPOINTS:
                    for num_samples in NUM_SAMPLES:
                        # # objects = wr.s3.list_objects(f's3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/{algo}/checkpoint_{checkpoint}/{vf}/{vf_checkpoint}/50_denoising_steps/{num_samples}_samples/*/results.txt')
                        # objects = wr.s3.list_objects(f's3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/{algo}/{checkpoint}/{vf}/{vf_checkpoint}/50_denoising_steps/{num_samples}_samples/*/results.txt')
                        # objects_json = wr.s3.list_objects(f's3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/{algo}/{checkpoint}/{vf}/{vf_checkpoint}/50_denoising_steps/{num_samples}_samples/*/results.json')
                        objects = wr.s3.list_objects(f's3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/{algo}/checkpoint_{checkpoint}/{vf}/checkpoint_{vf_checkpoint}/50_denoising_steps/{num_samples}_samples/tmpensb/*/results.txt')
                                                    #    s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_nam_20240201_014718/checkpoint_140000/no_vf/checkpoint_none/50_denoising_steps/1_samples/tmpensb/2024.02.22_00.07.46/results.txt
                        objects_json = wr.s3.list_objects(f's3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/{algo}/checkpoint_{checkpoint}/{vf}/checkpoint_{vf_checkpoint}/50_denoising_steps/{num_samples}_samples/tmpensb/*/results.json')
                        results_files += objects 
                        results_json_files += objects_json

    for results_file in sorted(results_files):
        print("\n" + results_file)
        with smart_open(results_file, 'rb') as s3_source:
            for line in s3_source:
                print(line.decode('utf8'), end="")


    for results_json in sorted(results_json_files):
        print("\n" + results_json)
        with smart_open(results_json, 'rb') as s3_source:
            for line in s3_source:
                # print(line.decode('utf8'), end="")
                s = line.decode('utf8')

            results = json.loads(s)
            

            for key, val in results["0"]["task_info"].items():
                success_rate = val["success"] / val["total"]
                print(f"{key}: {success_rate * 100:.1f}%")


if __name__ == "__main__":
    view_results()