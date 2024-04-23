#!/bin/bash

# Define your AWS S3 bucket name
bucket_name=$1


# Use the AWS CLI to list only files ending in ".tfrecord" in the bucket recursively and print only the last three elements of each file path
aws s3 ls "$bucket_name" --recursive | while read -r line; do
    # Extract the file path and check if it ends with ".tfrecord"
    file_path=$(echo "$line" | awk '{print $NF}')
    if [[ $file_path == *.tfrecord ]]; then
        reversed_path=$(echo "$file_path" | rev)
        last_three_elements=$(echo "$reversed_path" | awk -F'/' '{print $1 "/" $2 "/" $3}' | rev)
        echo "$last_three_elements"
    fi
done