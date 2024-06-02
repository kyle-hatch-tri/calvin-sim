#!/bin/bash

# # Define source and destination S3 URIs
# src_s3_bucket="s3://susie-data/libero_data_processed_atmsplithighlevel/"
# dest_s3_bucket="s3://susie-data/libero_data_processed_atmsplitlowlevel/"

# # Sync operation for 'train' directories - only copy traj files where X is 0-9
# aws s3 sync "$src_s3_bucket" "$dest_s3_bucket" \
#     --exclude '*' \
#     --include '*/train*/traj[0-9].tfrecord'

# # Sync operation for 'val' directories - copy all traj files
# aws s3 sync "$src_s3_bucket" "$dest_s3_bucket" \
#     --exclude '*' \
#     --exclude '*/train*/traj*.tfrecord' \
#     --include '*/val*/traj*.tfrecord'

# echo "S3 sync operations completed."









# Define source and destination S3 URIs
src_s3_bucket="s3://susie-data/libero_data_processed_split2/"
dest_s3_bucket="s3://susie-data/libero_data_processed_split210shot/"



# Sync all files from source to destination preserving the directory structure
aws s3 sync "$src_s3_bucket" "$dest_s3_bucket" 

# # Copy and rename files with 'val' in the path and X between 0-9
# aws s3 ls "$src_s3_bucket" --recursive | grep '/val.*/traj[0-9].tfrecord' | while read -r line; do
#     src_path=$(echo "$line" | awk '{print $4}')  # Extracts the full S3 path of the file
#     src_path="${src_path#*/}"
#     for i in {0..9}; do
#         new_path=$(echo "$src_path" | sed 's/val/train/')  # Replace 'val' with 'train' in the path
#         new_path=$(echo "$new_path" | sed -E "s/(traj[0-9]+)\.tfrecord/\1-${i}.tfrecord/")
#     # echo "${src_s3_bucket}${src_path} to ${dest_s3_bucket}${new_path}"
#     aws s3 cp "${src_s3_bucket}${src_path}" "${dest_s3_bucket}${new_path}"
#     done
    
# done

echo "S3 operations completed."
