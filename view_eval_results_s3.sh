cd /home/kylehatch/Desktop/hidql/eval_outputs
# aws s3 sync --exclude "*" --include "*results.txt" s3://kyle-sagemaker-training-outputs/eval-outputs/ .

IFS=$'\n'
paths=($(find . -name "*results.txt"))
unset IFS

printf "%s\n" "${paths[@]}"

for i in "${!low_level_checkpoints[@]}"
do
    echo "low_level_checkpoints[$i]: ${low_level_checkpoints[i]}"
    export CUDA_VISIBLE_DEVICES=$i 
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

done 