#!/bin/bash



srcdir=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/bridge_susie_checkpoints
s3_bucket=s3://kyle-sagemaker-training-outputs/susie_low_level/bridge

# find $srcdir -path "*checkpoint_150000/*"

checkpointpaths=($(find $srcdir -path "*/checkpoint"))

for checkpointpath in "${checkpointpaths[@]}"; do
#     echo "Processing checkpoint: $checkpointpath"
    rundir=$(echo "$checkpointpath" | awk -F'/' '{OFS="/"; $(NF-1)=""; NF-=2; print}')
    configpath="${rundir}/config.yaml"
#     echo "configpath: $configpath"

    last_half_of_path=$(echo "$configpath" | sed 's|.*bridge_susie_checkpoints/||')
    s3_uri=${s3_bucket}/${last_half_of_path}
#     echo $s3_uri
    aws s3 cp $s3_uri $rundir


    echo ""
done

