#!/bin/bash

export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"


srcdir=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/bridge_susie_checkpoints

checkpointpaths=($(find $srcdir -path "*checkpoint_150000/*"))

for checkpointpath in "${checkpointpaths[@]}"; do
    if [[ "$checkpointpath" == *"lcgcprogressvf"* ]]; then
            continue
    fi

    if [[ "$checkpointpath" == *"lcdiffusion"* ]]; then
            continue
    fi

    echo "Processing checkpoint: $checkpointpath"

    checkpointdir=$(echo "$checkpointpath" | awk -F'/' '{OFS="/"; NF--; print}')
    outpath="${checkpointdir}/serialized_policy_ckpt"
    echo "outpath: $outpath"

    rundir=$(echo "$checkpointpath" | awk -F'/' '{OFS="/"; $(NF-1)=""; NF-=2; print}')
    configpath="${rundir}/config.yaml"
    echo "configpath: $configpath"

    python3 -u scripts/policy_ckpt_to_hlo.py \
    --checkpoint_path $checkpointdir \
    --config_path $configpath \
    --outpath $outpath \
    --im_size 200

    echo ""
done