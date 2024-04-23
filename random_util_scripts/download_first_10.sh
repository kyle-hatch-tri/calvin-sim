#!/bin/bash

# AWS S3 Bucket name
bucket_name="s3://susie-data/calvin_data_processed/language_conditioned_16_samples"

dst_dir="/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_16_samples"

mkdir -p $dst_dir

# List of top-level subfolders
# top_subfolders=("training" "validation")
top_subfolders=("validation")



# Loop through each top-level subfolder
for top_subfolder in "${top_subfolders[@]}"; do
    # Determine the list of subfolders to iterate over
    if [ "$top_subfolder" == "validation" ]; then
        # Only loop through subfolder "D" for "validation" top_subfolder
        subfolders=("D")
        start=234
    elif [ "$top_subfolder" == "training" ]; then
        # List of subfolders within each top-level subfolder
        subfolders=("A" "B" "C" "D")
        start=0
    else
        echo "Unrecognized top_subfolder: $top_subfolder"
        exit 125
    fi


    
    # Loop through each subfolder within the top-level subfolder
    for subfolder in "${subfolders[@]}"; do
        # Loop through numbers 0 to 9
        # for i in {0..9}; do
        for ((i=start; i<start+10; i++)); do
            # File name
            file_name="traj${i}.tfrecord"
            
            # Download file from S3 bucket
            aws s3 cp "${bucket_name}/${top_subfolder}/${subfolder}/${file_name}" "${dst_dir}/${top_subfolder}/${subfolder}/"
        done
    done
done
