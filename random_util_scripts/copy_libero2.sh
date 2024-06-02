#!/bin/bash


# low level
suites=("libero_10" "libero_90" "libero_goal" "libero_object" "libero_spatial")

# Loop through the array
for suite in "${suites[@]}"; do
	srcdir=s3://susie-data/libero_data_processed_atmsplitlowlevel/$suite/train
	dstdir=s3://susie-data/libero_data_processed_atmsplitlowlevelall/train/$suite
	aws s3 sync $srcdir $dstdir

        
	srcdir=s3://susie-data/libero_data_processed_atmsplitlowlevel/$suite/val
	dstdir=s3://susie-data/libero_data_processed_atmsplitlowlevelall/val/$suite
	aws s3 sync $srcdir $dstdir
done



# high level 
suites=("libero_10" "libero_90" "libero_goal" "libero_object" "libero_spatial")

# Loop through the array
for suite in "${suites[@]}"; do
	srcdir=s3://susie-data/libero_data_processed_atmsplithighlevel/$suite/train
	dstdir=s3://susie-data/libero_data_processed_atmsplithighlevelall/train/$suite
	aws s3 sync $srcdir $dstdir
	
	srcdir=s3://susie-data/libero_data_processed_atmsplithighlevel/$suite/val
	dstdir=s3://susie-data/libero_data_processed_atmsplithighlevelall/val/$suite
	aws s3 sync $srcdir $dstdir
done


