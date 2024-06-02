#!/bin/bash



# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_10/train s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_10/train/libero_10
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_90/train s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_90/train/libero_90
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_goal/train s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_goal/train/libero_goal
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_object/train s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_object/train/libero_object
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_spatial/train s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_spatial/train/libero_spatial

# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_10/val s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_10/val/libero_10
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_90/val s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_90/val/libero_90
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_goal/val s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_goal/val/libero_goal
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_object/val s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_object/val/libero_object
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_spatial/val s3://susie-data/libero_data_processed_atmsplitlowlevel/libero_spatial/val/libero_spatial


# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_10/train s3://susie-data/libero_data_processed_atmsplithighlevel/libero_10/train/libero_10
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_90/train s3://susie-data/libero_data_processed_atmsplithighlevel/libero_90/train/libero_90
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_goal/train s3://susie-data/libero_data_processed_atmsplithighlevel/libero_goal/train/libero_goal
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_object/train s3://susie-data/libero_data_processed_atmsplithighlevel/libero_object/train/libero_object
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_spatial/train s3://susie-data/libero_data_processed_atmsplithighlevel/libero_spatial/train/libero_spatial

# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_10/val s3://susie-data/libero_data_processed_atmsplithighlevel/libero_10/val/libero_10
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_90/val s3://susie-data/libero_data_processed_atmsplithighlevel/libero_90/val/libero_90
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_goal/val s3://susie-data/libero_data_processed_atmsplithighlevel/libero_goal/val/libero_goal
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_object/val s3://susie-data/libero_data_processed_atmsplithighlevel/libero_object/val/libero_object
# aws s3 mv --recursive s3://susie-data/libero_data_processed_atmsplithighlevel/libero_spatial/val s3://susie-data/libero_data_processed_atmsplithighlevel/libero_spatial/val/libero_spatial



# low level
suites=("libero_10" "libero_90" "libero_goal" "libero_object" "libero_spatial")

# Loop through the array
for suite in "${suites[@]}"; do
	srcdir=s3://susie-data/libero_data_processed_atmsplitlowlevelall/train/$suite 
	dstdir=s3://susie-data/libero_data_processed_atmsplitlowlevel/$suite/train/$suite
	aws s3 sync $srcdir $dstdir

        
	srcdir=s3://susie-data/libero_data_processed_atmsplitlowlevelall/val/$suite 
	dstdir=s3://susie-data/libero_data_processed_atmsplitlowlevel/$suite/val/$suite
	aws s3 sync $srcdir $dstdir
done



# high level 
for suite in "${suites[@]}"; do
	srcdir=s3://susie-data/libero_data_processed_atmsplithighlevelall/train/$suite 
	dstdir=s3://susie-data/libero_data_processed_atmsplithighlevel/$suite/train/$suite
	aws s3 sync $srcdir $dstdir

        
	srcdir=s3://susie-data/libero_data_processed_atmsplithighlevelall/val/$suite 
	dstdir=s3://susie-data/libero_data_processed_atmsplithighlevel/$suite/val/$suite
	aws s3 sync $srcdir $dstdir
done