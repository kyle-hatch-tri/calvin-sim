import os 
from glob import glob 

"""
cd /home/kylehatch/Desktop/hidql/eval_outputs
aws s3 sync --exclude "*" --include "*results.txt" s3://kyle-sagemaker-training-outputs/eval-outputs/ .
"""

EVAL_OUTPUTS_DIR = "/home/kylehatch/Desktop/hidql/eval_outputs"

# ALGOS = ["public_checkpoint", "gcbc_diffusion", "gcbc", "gciql", "gciql2", "gciql3", "gciql4", "gciql5"]
# CHECKPOINTS = ["only_checkpoint", 5_000, 500_000, 1_000_000, 2_000_000]
# DENOISING_STEPS = [50]
# ENSEMBLE = ["tmpensb", "notmpensb"]

# ALGOS = ["public_checkpoint", "gcbc_diffusion"]
# CHECKPOINTS = ["only_checkpoint", 50_000, 500_000, 1_000_000, 2_000_000]
# DENOISING_STEPS = [50]
# ENSEMBLE = ["tmpensb", "notmpensb"]


ALGOS = ["gcbc", "gciql", "gciql2", "gciql3", "gciql4", "gciql5"]
CHECKPOINTS = ["only_checkpoint", 50_000]
DENOISING_STEPS = [50]
ENSEMBLE = ["notmpensb"]

# ALGOS = ["public_checkpoint", "gcbc_diffusion"]
# CHECKPOINTS = ["only_checkpoint", 50_000]
# DENOISING_STEPS = [50]
# ENSEMBLE = ["tmpensb"]


def print_results_file(results_file):
    print("\n" + results_file)
    with open(results_file, 'r') as f:
        print(f.read())


def view_outputs():
    filepaths = glob(os.path.join(EVAL_OUTPUTS_DIR, "**", "*results.txt"), recursive=True)
    for results_file in sorted(filepaths):
        results_file_parts = results_file.split("/")
        algo, checkpoint = results_file_parts[-5].split("-")
        # print(results_file)

        if "gcbc_diffusion" in algo:
            algo = "gcbc_diffusion"
        elif algo != "public_checkpoint":
            algo = algo.split("_")[0]

        if checkpoint != "only_checkpoint":
            checkpoint = int(checkpoint.split("_")[-1])

        denoising_steps = int(results_file_parts[-4].split("_")[0])
        ensemble = results_file_parts[-3]

        if algo not in ALGOS:
            continue 

        if checkpoint not in CHECKPOINTS:
            continue 

        if denoising_steps not in DENOISING_STEPS:
            continue 

        if ensemble not in ENSEMBLE:
            continue 


        print_results_file(results_file)
        
     


if __name__ == "__main__":
    view_outputs()