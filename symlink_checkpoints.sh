BASE_CHECKPOINT_DIR="/home/kylehatch/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix/snapshots/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"
TARGET_CHECKPOINT_DIR="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-80000"


# echo "Before"
# ls -l $TARGET_CHECKPOINT_DIR
for entry in "$BASE_CHECKPOINT_DIR"/*
do
  echo "$entry"
#   ln -s "$BASE_CHECKPOINT_DIR/$entry" "$TARGET_CHECKPOINT_DIR"
ln -s "$entry" "$TARGET_CHECKPOINT_DIR"
done

# echo "After"
# ls -l $TARGET_CHECKPOINT_DIR