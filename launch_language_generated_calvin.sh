#!/bin/bash

# Ensure arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 n_idxs n_gpus start_i s3"
    exit 1
fi

# Assign command-line arguments to variables
n_idxs=$1
n_gpus=$2
start_i=$3
s3=$4

# Perform assertion
if (( $n_idxs % $n_gpus != 0 )); then
    echo "Error: n_idxs is not divisible by n_gpus"
    exit 1
fi

# Calculate n_idxs_per_gpu
n_idxs_per_gpu=$((n_idxs / n_gpus))

# Initialize array to store start and end indices
declare -a start_and_end_idxs

# Loop through each GPU
for (( i=0; i<$n_gpus; i++ )); do 
    start_idx=$((start_i + i * n_idxs_per_gpu))
    end_idx=$((start_idx + n_idxs_per_gpu))
    # start_and_end_idxs+=("$s3 $i $start_idx $end_idx")

    export CUDA_VISIBLE_DEVICES=$i

    python3 -u language_conditioned_calvin_generated_goals.py \
    --start_i $start_idx \
    --end_i $end_idx \
    --s3 $s3 \
    --s3_dir "s3://susie-data/calvin_data_processed/language_conditioned_with_generated" &
done

wait 

echo "Done"
# # Print the result
# printf "%s\n" "${start_and_end_idxs[@]}"

# bash launch_language_generated_calvin.sh 3 3 0 1
# bash launch_language_generated_calvin.sh 2 2 0 0


# export CUDA_VISIBLE_DEVICES=0
# python3 -u language_conditioned_calvin_generated_goals.py \
# --start_i 0 \
# --end_i 2 \
# --s3 0 \
# --s3_dir "s3://susie-data/calvin_data_processed/language_conditioned_with_generated"